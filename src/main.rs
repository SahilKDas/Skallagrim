use skallagrim::{
    game::{Game, Move},
    neural::Models,
    search::{self, Bot, Limits},
};
use std::{
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};
fn emit(s: &str) {
    println!("{s}");
    let _ = io::stdout().flush();
}
fn stop_worker(worker: &mut Option<(Arc<AtomicBool>, thread::JoinHandle<()>)>) {
    if let Some((stop, handle)) = worker.take() {
        stop.store(true, Ordering::Relaxed);
        let _ = handle.join();
    }
}
fn position(parts: &[&str]) -> Result<Game, String> {
    let split = parts
        .iter()
        .position(|&s| s == "moves")
        .unwrap_or(parts.len());
    let mut g = match parts.first() {
        Some(&"startpos") if split == 1 => Game::default(),
        Some(&"fen") if split == 3 => Game::from_fen(&parts[1..3].join(" "))?,
        _ => return Err("position startpos|fen <board> <side> [moves ...]".into()),
    };
    if split < parts.len() {
        for m in &parts[split + 1..] {
            g.play(m.parse::<Move>()?)?;
        }
    }
    Ok(g)
}
fn main() {
    let mut game = Game::default();
    let mut bot = Bot::Baseline;
    let mut models = Models::default();
    let mut worker = None;
    for line in io::stdin().lock().lines() {
        let Ok(line) = line else {
            break;
        };
        let received = std::time::Instant::now();
        let p: Vec<_> = line.split_whitespace().collect();
        if p.is_empty() {
            continue;
        }
        if !matches!(p[0], "isready" | "uci" | "hti" | "stop") {
            stop_worker(&mut worker);
        }
        let result: Result<(), String> = (|| {
            match p[0] {
                "uci" | "hti" => {
                    emit(&format!("id name Skallagrim {}", env!("CARGO_PKG_VERSION")));
                    emit("id author Sahil K Das");
                    emit(
                        "option name Bot type combo default Baseline var Baseline var Rho var Iota var Beta",
                    );
                    emit("option name Variant type combo default Copenhagen var Copenhagen");
                    emit("option name NNUEFile type string default <empty>");
                    emit("option name CNNFile type string default <empty>");
                    emit(&format!(
                        "info string rules {}; conservative exit-fort detector; see docs/RULES.md",
                        skallagrim::game::RULES_VERSION
                    ));
                    emit("uciok");
                }
                "isready" => emit("readyok"),
                "ucinewgame" => game = Game::default(),
                "setoption" => {
                    let i = p
                        .iter()
                        .position(|&s| s == "value")
                        .ok_or("setoption needs value")?;
                    if p.get(1) != Some(&"name") || i < 3 {
                        return Err("setoption name <name> value <value>".into());
                    }
                    let name = p[2..i].join(" ");
                    let value = p[i + 1..].join(" ");
                    match name.as_str() {
                        "Bot" => bot = value.parse()?,
                        "Variant"
                            if value.eq_ignore_ascii_case("Copenhagen")
                                || value.eq_ignore_ascii_case("copenhagen11x11") => {}
                        "NNUEFile" => models.load_nnue(std::path::Path::new(&value))?,
                        "CNNFile" => models.load_cnn(std::path::Path::new(&value))?,
                        _ => return Err("unsupported option/value".into()),
                    }
                }
                "position" => game = position(&p[1..])?,
                "play" if p.len() == 2 => game.play(p[1].parse()?)?,
                "adjudicate" if p.len() == 2 => game.referee_decision(p[1])?,
                "prove" if p.get(1) == Some(&"draw") => {
                    let mut nodes = 10_000usize;
                    let mut milliseconds = 1_000u64;
                    if !(p.len() - 2).is_multiple_of(2) {
                        return Err("prove draw [nodes N] [movetime MS]".into());
                    }
                    for pair in p[2..].chunks(2) {
                        match pair[0] {
                            "nodes" => {
                                nodes = pair[1]
                                    .parse::<usize>()
                                    .map_err(|_| "invalid proof limit")?
                                    .min(100_000)
                            }
                            "movetime" => {
                                milliseconds = pair[1]
                                    .parse::<u64>()
                                    .map_err(|_| "invalid proof limit")?
                                    .min(60_000)
                            }
                            _ => return Err("unsupported proof limit".into()),
                        }
                    }
                    let g = game.clone();
                    let flag = Arc::new(AtomicBool::new(false));
                    let stop = flag.clone();
                    worker = Some((
                        flag,
                        thread::spawn(move || {
                            let proof = skallagrim::proof::prove_draw(
                                &g,
                                nodes,
                                Duration::from_millis(milliseconds),
                                &stop,
                            );
                            emit(&format!("proof {}", serde_json::to_string(&proof).unwrap()));
                        }),
                    ));
                }
                "snapshot" => emit(&format!("snapshot {}", game.snapshot())),
                "legal" => emit(&format!(
                    "legal {}",
                    game.legal_moves()
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(" ")
                )),
                "eval" => {
                    let acc = models.nnue.as_ref().map(|n| n.accumulator(&game));
                    emit(&format!(
                        "eval {}",
                        serde_json::json!({"features": skallagrim::neural::features(&game), "planes": skallagrim::neural::planes(&game), "nnue": acc.map(|a| models.nnue.as_ref().unwrap().value(&a)), "cnn": models.cnn.as_ref().map(|n| n.infer(&game))})
                    ));
                }
                "perft" if p.len() == 2 => emit(&format!(
                    "perft {}",
                    game.perft(p[1].parse().map_err(|_| "invalid depth")?)
                )),
                "go" => {
                    search::ready(bot, &models)?;
                    let mut limits = Limits::default();
                    if p.contains(&"infinite") {
                        limits.time = Duration::from_secs(86400);
                    }
                    let mut i = 1;
                    while i < p.len() {
                        match p[i] {
                            "movetime" | "nodes" | "depth" => {
                                if i + 1 >= p.len() {
                                    return Err("missing search limit".into());
                                }
                                let n: u64 =
                                    p[i + 1].parse().map_err(|_| "invalid search limit")?;
                                match p[i] {
                                    "movetime" => {
                                        limits.time = Duration::from_millis(n.min(86400000))
                                    }
                                    "nodes" => {
                                        limits.nodes = n;
                                        if !p.contains(&"movetime") {
                                            limits.time = Duration::from_secs(86400);
                                        }
                                    }
                                    _ => {
                                        limits.depth = n.min(64) as u32;
                                        if !p.contains(&"movetime") {
                                            limits.time = Duration::from_secs(86400);
                                        }
                                    }
                                };
                                i += 2;
                            }
                            "infinite" => i += 1,
                            _ => return Err("unsupported go argument".into()),
                        }
                    }
                    let deadline = search::response_deadline(received, limits.time);
                    let g = game.clone();
                    let models = models.clone();
                    let stop = Arc::new(AtomicBool::new(false));
                    let flag = stop.clone();
                    worker = Some((
                        stop,
                        thread::spawn(move || {
                            let side_sign = g.side.sign();
                            search::search_with_deadline(
                                g,
                                bot,
                                models,
                                limits,
                                flag,
                                deadline,
                                |result| match result {
                                    Ok(r) => {
                                        if r.deadline_fallback {
                                            emit("info string deadline-fallback");
                                        }
                                        emit(&format!(
                                            "info depth {} score cp {} nodes {} time {} pv {}",
                                            r.depth,
                                            (r.value * side_sign * 1000.0) as i32,
                                            r.nodes,
                                            r.elapsed.as_millis(),
                                            r.best.map(|m| m.to_string()).unwrap_or_default()
                                        ));
                                        emit(&format!(
                                            "info string policy {}",
                                            serde_json::to_string(
                                                &r.policy
                                                    .iter()
                                                    .map(|(m, p)| (m.to_string(), *p))
                                                    .collect::<Vec<_>>()
                                            )
                                            .unwrap()
                                        ));
                                        emit(&format!(
                                            "bestmove {}",
                                            r.best
                                                .map(|m| m.to_string())
                                                .unwrap_or_else(|| "0000".into())
                                        ));
                                    }
                                    Err(e) => {
                                        emit(&format!("info string error {e}"));
                                        emit("bestmove 0000");
                                    }
                                },
                            );
                        }),
                    ));
                }
                "stop" => stop_worker(&mut worker),
                "quit" => {}
                _ => return Err("unknown command".into()),
            }
            Ok(())
        })();
        if let Err(e) = result {
            emit(&format!("info string error {e}"));
        }
        if p[0] == "quit" {
            break;
        }
    }
    stop_worker(&mut worker);
}
