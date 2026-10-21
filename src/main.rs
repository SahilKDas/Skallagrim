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
fn search_reply(r: &search::SearchResult, side_sign: f32) -> String {
    let fallback = if r.deadline_fallback {
        "info string deadline-fallback\n"
    } else {
        ""
    };
    let best = r
        .best
        .map(|m| m.to_string())
        .unwrap_or_else(|| "0000".into());
    let policy = serde_json::to_string(
        &r.policy
            .iter()
            .map(|(m, p)| (m.to_string(), *p))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    format!(
        "{fallback}info depth {} score cp {} nodes {} time {} pv {}\ninfo string policy {}\nbestmove {}\n",
        r.depth,
        (r.value * side_sign * 1000.0) as i32,
        r.nodes,
        r.elapsed.as_millis(),
        r.best.map(|m| m.to_string()).unwrap_or_default(),
        policy,
        best
    )
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
                    let started = std::time::Instant::now();
                    let nnue = models.nnue.as_ref().map(|n| n.value(&n.accumulator(&game)));
                    let nnue_ns = nnue.map(|_| started.elapsed().as_nanos());
                    let started = std::time::Instant::now();
                    let cnn = models.cnn.as_ref().map(|n| n.infer(&game));
                    let cnn_ns = cnn.as_ref().map(|_| started.elapsed().as_nanos());
                    emit(&format!(
                        "eval {}",
                        serde_json::json!({"features": skallagrim::neural::features(&game), "nnue_features": models.nnue.as_ref().map(|n| n.active(&game)), "planes": skallagrim::neural::planes(&game), "nnue": nnue, "cnn": cnn, "timing": {"nnue_rebuild_value_ns": nnue_ns, "cnn_inference_ns": cnn_ns}})
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
                    let trace_timing = std::env::var_os("SKALLAGRIM_TRACE_TIMING").is_some();
                    let g = game.clone();
                    let models = models.clone();
                    let stop = Arc::new(AtomicBool::new(false));
                    let flag = stop.clone();
                    worker = Some((
                        stop,
                        thread::spawn(move || {
                            let entered_us = received.elapsed().as_micros();
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
                                        let publication_us = received.elapsed().as_micros();
                                        let reply = search_reply(&r, side_sign);
                                        let mut stdout = io::stdout().lock();
                                        let _ = stdout.write_all(reply.as_bytes());
                                        let _ = stdout.flush();
                                        drop(stdout);
                                        if trace_timing {
                                            eprintln!(
                                                "{{\"entered_us\":{entered_us},\"publication_us\":{publication_us},\"flushed_us\":{},\"search_us\":{},\"fallback\":{}}}",
                                                received.elapsed().as_micros(),
                                                r.elapsed.as_micros(),
                                                r.deadline_fallback
                                            );
                                        }
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

#[cfg(test)]
mod reply_tests {
    use super::*;
    #[test]
    fn batched_reply_preserves_policy_before_exactly_one_bestmove() {
        let mv = Game::default().legal_moves()[0];
        let result = search::SearchResult {
            best: Some(mv),
            value: 0.5,
            nodes: 42,
            depth: 2,
            elapsed: Duration::from_millis(240),
            policy: vec![(mv, 1.0)],
            deadline_fallback: true,
        };
        let reply = search_reply(&result, -1.0);
        let lines: Vec<_> = reply.lines().collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[0], "info string deadline-fallback");
        assert!(lines[1].contains("score cp -500"));
        assert!(lines[2].starts_with("info string policy "));
        assert_eq!(lines[3], format!("bestmove {mv}"));
        assert_eq!(reply.matches("bestmove ").count(), 1);
    }
}
