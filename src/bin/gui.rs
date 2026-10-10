use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use skallagrim::{
    game::{Game, Move, Side},
    neural::Models,
    render::{Layout, Renderer, View},
    search::{self, Bot, Limits},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let working = std::env::current_dir()?;
    let data_root = if let Some(i) = args.iter().position(|s| s == "--data-dir") {
        std::path::PathBuf::from(args.get(i + 1).ok_or("--data-dir needs path")?)
    } else if working.join("models/nnue.json").exists() || working.join("models/cnn.json").exists()
    {
        working.clone()
    } else {
        std::env::current_exe()?
            .ancestors()
            .skip(1)
            .find(|p| p.join("models/nnue.json").exists() || p.join("models/cnn.json").exists())
            .map(std::path::Path::to_path_buf)
            .unwrap_or(working)
    };
    let renderer = Renderer::default();
    let mut models = Models::default();
    if data_root.join("models/nnue.json").exists() {
        models.load_nnue(&data_root.join("models/nnue.json"))?;
    }
    if data_root.join("models/cnn.json").exists() {
        models.load_cnn(&data_root.join("models/cnn.json"))?;
    }
    let mut game = Game::default();
    let mut bot = if models.nnue.is_some() {
        Bot::Iota
    } else {
        Bot::Baseline
    };
    let mut human = Some(Side::Defender);
    let mut states = vec![game.clone()];
    let mut moves = vec![];
    if let Some(i) = args.iter().position(|s| s == "--replay") {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(
            args.get(i + 1).ok_or("--replay needs path")?,
        )?)?;
        game = Game::from_fen(
            value
                .get("start")
                .and_then(|v| v.as_str())
                .unwrap_or(skallagrim::game::START),
        )?;
        states = vec![game.clone()];
        if let Some(opening) = value
            .get("opening")
            .and_then(|v| v.get("moves"))
            .and_then(|v| v.as_array())
        {
            for item in opening {
                let text = item.as_str().ok_or("invalid opening move")?;
                game.play(text.parse()?)?;
                moves.push(text.to_string());
                states.push(game.clone());
            }
        }
        for item in value
            .get("moves")
            .and_then(|v| v.as_array())
            .ok_or("replay requires moves")?
        {
            let text = item
                .as_str()
                .or_else(|| item.get("move").and_then(|v| v.as_str()))
                .ok_or("invalid replay move")?;
            game.play(text.parse()?)?;
            moves.push(text.to_string());
            states.push(game.clone());
        }
        if game.outcome.is_none() {
            if let Some(outcome) = value.get("outcome").filter(|v| !v.is_null()) {
                game.outcome = Some(serde_json::from_value(outcome.clone())?);
            } else if let Some(reason) = value.get("reason").and_then(|v| v.as_str()) {
                game.outcome = Some(skallagrim::game::Outcome {
                    winner: serde_json::from_value(
                        value
                            .get("winner")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                    )?,
                    reason: reason.to_string(),
                });
            }
            if let Some(state) = states.last_mut() {
                *state = game.clone();
            }
        }
        human = None;
    }
    let mut selected = None;
    let mut cursor = 60;
    let mut status =
        "300 ms/move • Arrows + Enter to play • Left/Right with Alt to replay".to_string();
    if let Some(i) = args.iter().position(|s| s == "--render") {
        let pix = renderer.draw(
            1180,
            880,
            View {
                game: &game,
                selected: None,
                hover: None,
                cursor,
                bot: if bot == Bot::Iota {
                    "Iota / NNUE + alpha-beta"
                } else {
                    "Baseline (handcrafted)"
                },
                mode: "Human defenders",
                status: &status,
                moves: &moves,
                thinking: false,
                mouse: None,
                rules_open: false,
            },
        );
        pix.save_png(args.get(i + 1).ok_or("--render needs path")?)?;
        return Ok(());
    }
    let mut window = Window::new(
        "Skallagrim · Copenhagen Hnefatafl",
        1180,
        880,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(60);
    let (tx, rx) = mpsc::channel();
    let mut running: Option<(Arc<AtomicBool>, u64)> = None;
    let mut generation = 0u64;
    let mut down = false;
    let mut replay = None;
    let mut rules_open = false;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        while let Ok((id, result)) = rx.try_recv() {
            if id != generation {
                continue;
            }
            running = None;
            match result {
                Ok(r) => {
                    let r: search::SearchResult = r;
                    status = format!(
                        "{:?} · {} nodes · {} ms · attacker value {:.3}",
                        bot,
                        r.nodes,
                        r.elapsed.as_millis(),
                        r.value
                    );
                    if let Some(m) = r.best
                        && game.play(m).is_ok()
                    {
                        moves.push(m.to_string());
                        states.push(game.clone());
                    }
                }
                Err(e) => status = e,
            }
        }
        let (w, h) = window.get_size();
        let layout = Layout::new(w as u32, h as u32);
        let mouse = window.get_mouse_pos(MouseMode::Discard);
        let hover = mouse.and_then(|(x, y)| layout.square(x, y));
        let clicked = window.get_mouse_down(MouseButton::Left) && !down;
        down = window.get_mouse_down(MouseButton::Left);
        let panel_button = if clicked {
            mouse.and_then(|(x, y)| {
                if x >= layout.panel + 16.0 && x <= w as f32 - 44.0 {
                    [242.0, 286.0, 330.0, 374.0, 418.0, 462.0]
                        .iter()
                        .position(|a| y >= *a && y < *a + 34.0)
                } else {
                    None
                }
            })
        } else {
            None
        };
        let cancel = |running: &mut Option<(Arc<AtomicBool>, u64)>| {
            if let Some((s, _)) = running.take() {
                s.store(true, Ordering::Relaxed);
            }
        };
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            cancel(&mut running);
            generation += 1;
            rules_open = !rules_open;
        }
        if window.is_key_pressed(Key::N, KeyRepeat::No) || panel_button == Some(0) {
            cancel(&mut running);
            generation += 1;
            game = Game::default();
            states = vec![game.clone()];
            moves.clear();
            selected = None;
            replay = None;
        }
        if window.is_key_pressed(Key::U, KeyRepeat::No) || panel_button == Some(1) {
            cancel(&mut running);
            generation += 1;
            if states.len() > 1 {
                states.pop();
                moves.pop();
                game = states.last().unwrap().clone();
                if human.is_some_and(|side| side != game.side) && states.len() > 1 {
                    states.pop();
                    moves.pop();
                    game = states.last().unwrap().clone();
                }
            }
            selected = None;
            replay = None;
        }
        if window.is_key_pressed(Key::B, KeyRepeat::No) || panel_button == Some(3) {
            cancel(&mut running);
            generation += 1;
            bot = match bot {
                Bot::Baseline => Bot::Rho,
                Bot::Rho => Bot::Iota,
                Bot::Iota => Bot::Beta,
                Bot::Beta => Bot::Baseline,
            };
            status = search::ready(bot, &models)
                .err()
                .unwrap_or_else(|| format!("{:?} selected", bot));
        }
        if window.is_key_pressed(Key::H, KeyRepeat::No) || panel_button == Some(4) {
            cancel(&mut running);
            generation += 1;
            human = match human {
                Some(Side::Defender) => Some(Side::Attacker),
                Some(Side::Attacker) => None,
                None => Some(Side::Defender),
            };
        }
        if replay.is_none()
            && (window.is_key_pressed(Key::D, KeyRepeat::No)
                || window.is_key_pressed(Key::F, KeyRepeat::No))
        {
            cancel(&mut running);
            generation += 1;
            let decision = if window.is_key_pressed(Key::D, KeyRepeat::No) {
                "draw"
            } else {
                "fort"
            };
            status = match game.referee_decision(decision) {
                Ok(()) => format!("Referee adjudicated {decision}"),
                Err(e) => e,
            };
            if let Some(state) = states.last_mut() {
                *state = game.clone();
            }
        }
        if window.is_key_pressed(Key::S, KeyRepeat::No) || panel_button == Some(5) {
            std::fs::create_dir_all(data_root.join("runs"))?;
            let file = format!(
                "runs/replay-{}.json",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_millis()
            );
            std::fs::write(
                data_root.join(&file),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"format":"skallagrim-replay-v1", "start": skallagrim::game::START, "moves": moves, "outcome":game.outcome}),
                )?,
            )?;
            status = format!("Saved {file}");
        }
        if window.is_key_down(Key::LeftAlt) || window.is_key_down(Key::RightAlt) {
            if window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
                replay = Some(replay.unwrap_or(states.len() - 1).saturating_sub(1));
            }
            if window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
                let r = (replay.unwrap_or(states.len() - 1) + 1).min(states.len() - 1);
                replay = if r == states.len() - 1 { None } else { Some(r) };
            }
        } else {
            if window.is_key_pressed(Key::Left, KeyRepeat::Yes) && cursor % 11 > 0 {
                cursor -= 1;
            }
            if window.is_key_pressed(Key::Right, KeyRepeat::Yes) && cursor % 11 < 10 {
                cursor += 1;
            }
            if window.is_key_pressed(Key::Up, KeyRepeat::Yes) && cursor / 11 < 10 {
                cursor += 11;
            }
            if window.is_key_pressed(Key::Down, KeyRepeat::Yes) && cursor / 11 > 0 {
                cursor -= 11;
            }
        }
        let activated = if clicked {
            hover
        } else if window.is_key_pressed(Key::Enter, KeyRepeat::No) {
            Some(cursor)
        } else {
            None
        };
        if !rules_open
            && running.is_none()
            && replay.is_none()
            && (human.is_none() || human == Some(game.side))
            && let Some(s) = activated
        {
            cursor = s;
            if let Some(from) = selected {
                let m = Move { from, to: s };
                if game.play(m).is_ok() {
                    moves.push(m.to_string());
                    states.push(game.clone());
                    selected = None;
                } else {
                    selected = if game.board[s].side() == Some(game.side) {
                        Some(s)
                    } else {
                        None
                    };
                }
            } else if game.board[s].side() == Some(game.side) {
                selected = Some(s);
            }
        }
        let request = window.is_key_pressed(Key::Space, KeyRepeat::No) || panel_button == Some(2);
        if !rules_open
            && running.is_none()
            && replay.is_none()
            && game.outcome.is_none()
            && (request || human.is_some_and(|s| s != game.side))
        {
            match search::ready(bot, &models) {
                Err(e) => status = e,
                Ok(()) => {
                    let flag = Arc::new(AtomicBool::new(false));
                    let stop = flag.clone();
                    let g = game.clone();
                    let m = models.clone();
                    let tx = tx.clone();
                    let id = generation;
                    running = Some((flag, id));
                    std::thread::spawn(move || {
                        let _ = tx.send((
                            id,
                            search::search(
                                &g,
                                bot,
                                &m,
                                Limits {
                                    time: Duration::from_millis(300),
                                    ..Limits::default()
                                },
                                stop,
                            ),
                        ));
                    });
                }
            }
        }
        let mode = match human {
            Some(Side::Defender) => "Human defenders · engine attackers",
            Some(Side::Attacker) => "Human attackers · engine defenders",
            None => "Two humans · Space requests engine",
        };
        let display = replay.map(|r| &states[r]).unwrap_or(&game);
        let pix = renderer.draw(
            w.max(1) as u32,
            h.max(1) as u32,
            View {
                game: display,
                selected,
                hover,
                cursor,
                bot: match bot {
                    Bot::Baseline => "Baseline (handcrafted)",
                    Bot::Rho => "Rho / CNN + MCTS",
                    Bot::Iota => "Iota / NNUE + alpha-beta",
                    Bot::Beta => "Beta / NNUE + MCTS",
                },
                mode,
                status: &status,
                moves: &moves[..replay.unwrap_or(moves.len())],
                thinking: running.is_some(),
                mouse,
                rules_open,
            },
        );
        let buffer: Vec<_> = pix
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | p[2] as u32)
            .collect();
        window.update_with_buffer(&buffer, w.max(1), h.max(1))?;
    }
    if let Some((s, _)) = running {
        s.store(true, Ordering::Relaxed);
    }
    Ok(())
}
