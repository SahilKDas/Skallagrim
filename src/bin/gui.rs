use minifb::{InputCallback, Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use skallagrim::{
    game::{Move, Side},
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
struct NoteInput(std::rc::Rc<std::cell::RefCell<Vec<u32>>>);
fn replay_analysis(value: &serde_json::Value) -> Option<&serde_json::Value> {
    for event in value["referee"]["events"].as_array()?.iter().rev() {
        match event["kind"].as_str()? {
            "analysis" => return Some(&event["evidence"]),
            "ruling" if event["evidence"]["analysis"].is_object() => {
                return Some(&event["evidence"]["analysis"]);
            }
            "move" | "rewind" | "start" => return None,
            _ => {}
        }
    }
    None
}
impl InputCallback for NoteInput {
    fn add_char(&mut self, value: u32) {
        self.0.borrow_mut().push(value);
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let presence = skallagrim::presence::Presence::start(
        !args.iter().any(|s| s == "--no-discord" || s == "--render"),
        args.iter()
            .position(|s| s == "--discord-app-id")
            .and_then(|i| args.get(i + 1))
            .cloned()
            .or_else(|| std::env::var("SKALLAGRIM_DISCORD_APP_ID").ok()),
    );
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
    let mut referee =
        skallagrim::referee_client::RefereeClient::spawn(&data_root.join("runs/referee"))?;
    let mut game = referee.game()?;
    let mut bot = if models.nnue.is_some() {
        Bot::Iota
    } else {
        Bot::Baseline
    };
    let mut human = Some(Side::Defender);
    let mut states = vec![game.clone()];
    let mut moves = vec![];
    let mut loaded_replay = None;
    if let Some(i) = args.iter().position(|s| s == "--replay") {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(
            args.get(i + 1).ok_or("--replay needs path")?,
        )?)?;
        let loaded = skallagrim::replay::Replay::from_value(&value)?;
        loaded_replay = Some(value);
        states = loaded.states;
        moves = loaded.moves;
        game = states.last().unwrap().clone();
        human = None;
    }
    referee.call(serde_json::json!({"command":"new","fen":states[0].fen(),"moves":moves}))?;
    let mut referee_open = args.iter().any(|s| s == "--referee-panel");
    let mut pending: Option<&str> = None;
    let mut pending_revision = None;
    let mut note = String::new();
    let mut note_focus = false;
    let mut evidence_scroll = 0usize;
    let mut proof_preview: Option<(u64, usize, skallagrim::game::Game)> = None;
    let operator = args
        .iter()
        .position(|s| s == "--referee-name")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("Local GUI human");
    let mut evidence = vec!["No qualitative certificate yet.".to_string()];
    let mut selected = None;
    let mut cursor = 60;
    let mut status = format!("300 ms/move • {}", presence.hint);
    if let Some(i) = args.iter().position(|s| s == "--render") {
        if let Some(value) = loaded_replay.as_ref().and_then(replay_analysis) {
            evidence = vec![
                format!(
                    "{}: {}",
                    value["claim"].as_str().unwrap_or("analysis"),
                    value["status"].as_str().unwrap_or("unknown")
                ),
                value["reason"].as_str().unwrap_or("").to_string(),
            ];
            if let Some(nodes) = value["proof"]["certificate"]["nodes"].as_array() {
                evidence.push(format!("Verified strategy: {} nodes", nodes.len()));
                evidence.push("PgUp/PgDn: nodes; Home: live board".into());
                evidence.push(format!(
                    "Protected region: {}",
                    value["proof"]["certificate"]["region"]
                ));
                for edge in nodes[0]["edges"].as_array().into_iter().flatten().take(8) {
                    evidence.push(format!(
                        "{} -> node {}",
                        edge["mv"].as_str().unwrap_or(""),
                        edge["next"]
                    ));
                }
            }
        }
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
                mode: if human.is_none() {
                    "Replay · N starts a new game"
                } else {
                    "Human defenders · engine attackers"
                },
                status: &status,
                moves: &moves,
                thinking: false,
                mouse: None,
                rules_open: false,
                referee: referee_open.then_some((&evidence, &note, pending, evidence_scroll)),
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
    let note_input = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    window.set_input_callback(Box::new(NoteInput(note_input.clone())));
    let mut last_poll = std::time::Instant::now();
    let (tx, rx) = mpsc::channel();
    let mut running: Option<(Arc<AtomicBool>, u64)> = None;
    let mut generation = 0u64;
    let mut down = false;
    let mut replay = args
        .iter()
        .any(|s| s == "--replay")
        .then_some(states.len() - 1);
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
                    if let Some(m) = r.best {
                        match referee.play(m) {
                            Ok(next) => {
                                game = next;
                                moves.push(m.to_string());
                                states.push(game.clone());
                            }
                            Err(error) => {
                                game = referee.game()?;
                                status = error;
                            }
                        }
                    }
                }
                Err(e) => status = e,
            }
        }
        if last_poll.elapsed() >= Duration::from_millis(50) {
            presence.update(
                if replay.is_some() || loaded_replay.is_some() {
                    "Replay"
                } else if human.is_none() {
                    "Bot game"
                } else {
                    "Human vs bot"
                },
                &format!("{bot:?}"),
                if game.side == Side::Attacker {
                    "Attackers"
                } else {
                    "Defenders"
                },
                game.outcome.as_ref().map(|o| o.reason.as_str()),
            );
            referee.call(serde_json::json!({"command":"snapshot"}))?;
            if replay.is_none() && loaded_replay.is_none() {
                game = referee.game()?;
            }
            if game.outcome.is_some()
                && let Some((flag, _)) = running.take()
            {
                flag.store(true, Ordering::Relaxed);
            }
            if replay.is_none()
                && loaded_replay.is_none()
                && let Some(last) = states.last_mut()
            {
                *last = game.clone();
            }
            evidence = vec![
                format!(
                    "Turn: {:?}; revision {}",
                    game.side, referee.state["revision"]
                ),
                format!(
                    "Clock: {}",
                    if referee.state["clock"].is_null() {
                        "paused / untimed".to_string()
                    } else {
                        format!("{} ms left", referee.state["clock"]["remaining_ms"])
                    }
                ),
                format!(
                    "Result: {}",
                    game.outcome
                        .as_ref()
                        .map(|o| o.reason.as_str())
                        .unwrap_or("ongoing")
                ),
            ];
            let value = loaded_replay
                .as_ref()
                .and_then(replay_analysis)
                .unwrap_or(&referee.state["analysis"]);
            if value.is_object() {
                evidence.extend([
                    format!(
                        "{}: {}",
                        value["claim"].as_str().unwrap_or("analysis"),
                        value["status"].as_str().unwrap_or("unknown")
                    ),
                    value["reason"]
                        .as_str()
                        .or_else(|| value["proof"]["reason"].as_str())
                        .unwrap_or("")
                        .to_string(),
                    format!("Positions: {}", value["proof"]["positions"]),
                    format!("Witness: {}", value["proof"]["witness"]),
                ]);
                if let Some(nodes) = value["proof"]["certificate"]["nodes"].as_array() {
                    evidence.push(format!(
                        "Verified strategy: {} nodes; protected squares {}",
                        nodes.len(),
                        value["proof"]["certificate"]["region"]
                    ));
                    evidence.push("PgUp/PgDn: nodes; Home: live board".into());
                    for (i, node) in nodes.iter().take(100).enumerate() {
                        evidence.push(format!("Node {i}: {}", node["fen"].as_str().unwrap_or("")));
                        if let Some(edges) = node["edges"].as_array() {
                            for edge in edges {
                                evidence.push(format!(
                                    "  {} -> node {}",
                                    edge["mv"].as_str().unwrap_or(""),
                                    edge["next"]
                                ));
                            }
                        }
                    }
                    if nodes.len() > 100 {
                        evidence.push(
                            "Panel shows first 100 nodes; full certificate is in the replay log."
                                .into(),
                        );
                    }
                }
            }
            let ruling = loaded_replay
                .as_ref()
                .and_then(|v| v["referee"]["events"].as_array())
                .and_then(|events| events.iter().rev().find(|e| e["kind"] == "ruling"))
                .unwrap_or(&referee.state["last_ruling"]);
            if ruling.is_object() {
                evidence.push(format!(
                    "Ruling: {} / {}",
                    ruling["evidence"]["source"], ruling["evidence"]["reason"]
                ));
                if let Some(operator) = ruling["evidence"]["operator"].as_str() {
                    evidence.push(format!("Human: {operator}"));
                }
            }
            last_poll = std::time::Instant::now();
        }
        for code in note_input.borrow_mut().drain(..) {
            if note_focus
                && let Some(ch) = char::from_u32(code)
                && !ch.is_control()
                && note.chars().count() < 500
            {
                note.push(ch);
            }
        }
        if note_focus && window.is_key_pressed(Key::Backspace, KeyRepeat::Yes) {
            note.pop();
        }
        let (w, h) = window.get_size();
        let layout = Layout::new(w as u32, h as u32);
        let mouse = window.get_mouse_pos(MouseMode::Discard);
        if referee_open
            && mouse.is_some_and(|(x, _)| x >= layout.panel)
            && let Some((_, scroll)) = window.get_scroll_wheel()
        {
            evidence_scroll = if scroll > 0.0 {
                evidence_scroll.saturating_sub(2)
            } else if scroll < 0.0 {
                {
                    let columns = ((w as f32 - layout.panel - 68.0)
                        / (7.0 * skallagrim::render::referee_scale(h as u32)).max(5.5))
                    .max(10.0) as usize;
                    let lines: usize = evidence
                        .iter()
                        .map(|line| line.chars().count().div_ceil(columns))
                        .sum();
                    (evidence_scroll + 2).min(lines.saturating_sub(8))
                }
            } else {
                evidence_scroll
            };
        }
        let hover = mouse.and_then(|(x, y)| layout.square(x, y));
        let clicked = window.get_mouse_down(MouseButton::Left) && !down;
        down = window.get_mouse_down(MouseButton::Left);
        let panel_button = if clicked && !referee_open {
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
        let cancel = |running: &mut Option<(Arc<AtomicBool>, u64)>,
                      referee: &mut skallagrim::referee_client::RefereeClient|
         -> Result<(), String> {
            if let Some((s, _)) = running.take() {
                s.store(true, Ordering::Relaxed);
            }
            referee.call(serde_json::json!({"command":"pause","reason":"Human canceled GUI search or changed view"}))?;
            Ok(())
        };
        if !note_focus && window.is_key_pressed(Key::J, KeyRepeat::No) {
            cancel(&mut running, &mut referee)?;
            generation += 1;
            referee.call(
                serde_json::json!({"command":"pause","reason":"human opened referee panel"}),
            )?;
            referee_open = !referee_open;
            evidence_scroll = 0;
        }
        if referee_open
            && clicked
            && let Some((x, y)) = mouse
            && x >= layout.panel + 16.0
            && x <= w as f32 - 44.0
        {
            let y = 98.0 + (y - 98.0) / skallagrim::render::referee_scale(h as u32);
            note_focus = (555.0..620.0).contains(&y);
            if (208.0..242.0).contains(&y) || (248.0..282.0).contains(&y) {
                let claim = if y < 242.0 { "draw" } else { "fort" };
                status = match referee.call(serde_json::json!({"command":"analyze","claim":claim,"nodes":10000,"milliseconds":500})) { Ok(_)=>format!("Analyzing {claim}; clocks remain authoritative"),Err(e)=>e };
            } else if (465.0..499.0).contains(&y) {
                pending = Some("draw");
                pending_revision = referee.state["revision"].as_u64();
            } else if (505.0..539.0).contains(&y) {
                pending = Some("fort");
                pending_revision = referee.state["revision"].as_u64();
            } else if (642.0..676.0).contains(&y)
                && replay.is_none()
                && loaded_replay.is_none()
                && let Some(claim) = pending
            {
                status = match referee.call(serde_json::json!({"command":"adjudicate","claim":claim,"operator":operator,"reason":note,"revision":pending_revision})) { Ok(_)=>{game=referee.game()?; pending=None; "Human ruling recorded with evidence".into()},Err(e)=>e };
            } else if (688.0..722.0).contains(&y) {
                pending = None;
                note.clear();
            } else if (740.0..774.0).contains(&y) {
                referee_open = false;
                note_focus = false;
            }
        }
        if !note_focus && window.is_key_pressed(Key::R, KeyRepeat::No) {
            cancel(&mut running, &mut referee)?;
            generation += 1;
            rules_open = !rules_open;
        }
        if !note_focus && window.is_key_pressed(Key::N, KeyRepeat::No) || panel_button == Some(0) {
            cancel(&mut running, &mut referee)?;
            generation += 1;
            referee.call(serde_json::json!({"command":"new"}))?;
            game = referee.game()?;
            states = vec![game.clone()];
            moves.clear();
            loaded_replay = None;
            pending = None;
            note.clear();
            note_focus = false;
            evidence_scroll = 0;
            selected = None;
            replay = None;
        }
        if !note_focus && window.is_key_pressed(Key::U, KeyRepeat::No) || panel_button == Some(1) {
            cancel(&mut running, &mut referee)?;
            generation += 1;
            let before = moves.len();
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
            if before > moves.len() {
                loaded_replay = None;
                referee.call(serde_json::json!({"command":"undo","count":before-moves.len()}))?;
                game = referee.game()?;
                pending = None;
                note.clear();
                note_focus = false;
            }
            selected = None;
            replay = None;
        }
        if !note_focus && window.is_key_pressed(Key::B, KeyRepeat::No) || panel_button == Some(3) {
            cancel(&mut running, &mut referee)?;
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
        if !note_focus && window.is_key_pressed(Key::H, KeyRepeat::No) || panel_button == Some(4) {
            cancel(&mut running, &mut referee)?;
            generation += 1;
            human = match human {
                Some(Side::Defender) => Some(Side::Attacker),
                Some(Side::Attacker) => None,
                None => Some(Side::Defender),
            };
        }
        if !note_focus
            && replay.is_none()
            && loaded_replay.is_none()
            && (window.is_key_pressed(Key::D, KeyRepeat::No)
                || window.is_key_pressed(Key::F, KeyRepeat::No))
        {
            cancel(&mut running, &mut referee)?;
            generation += 1;
            let decision = if !note_focus && window.is_key_pressed(Key::D, KeyRepeat::No) {
                "draw"
            } else {
                "fort"
            };
            referee.call(
                serde_json::json!({"command":"pause","reason":"human reviewing qualitative claim"}),
            )?;
            referee_open = true;
            pending = Some(decision);
            pending_revision = referee.state["revision"].as_u64();
            note_focus = false;
            status = "Review evidence and enter a reason before confirming.".into();
        }
        if !note_focus && window.is_key_pressed(Key::S, KeyRepeat::No) || panel_button == Some(5) {
            std::fs::create_dir_all(data_root.join("runs"))?;
            let file = format!(
                "runs/replay-{}.json",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_millis()
            );
            std::fs::write(
                data_root.join(&file),
                serde_json::to_vec_pretty(&if let Some(value) = &loaded_replay {
                    value.clone()
                } else {
                    referee.call(serde_json::json!({"command":"export"}))?
                })?,
            )?;
            status = format!("Saved {file}");
        }
        if window.is_key_down(Key::LeftAlt) || window.is_key_down(Key::RightAlt) {
            if !note_focus && window.is_key_pressed(Key::Left, KeyRepeat::Yes)
                || window.is_key_pressed(Key::Right, KeyRepeat::Yes)
            {
                cancel(&mut running, &mut referee)?;
                generation += 1;
                selected = None;
            }
            if !note_focus && window.is_key_pressed(Key::Left, KeyRepeat::Yes) {
                replay = Some(replay.unwrap_or(states.len() - 1).saturating_sub(1));
            }
            if !note_focus && window.is_key_pressed(Key::Right, KeyRepeat::Yes) {
                let r = (replay.unwrap_or(states.len() - 1) + 1).min(states.len() - 1);
                replay = if r == states.len() - 1 && loaded_replay.is_none() {
                    None
                } else {
                    Some(r)
                };
            }
        } else {
            if !note_focus && window.is_key_pressed(Key::Left, KeyRepeat::Yes) && cursor % 11 > 0 {
                cursor -= 1;
            }
            if !note_focus && window.is_key_pressed(Key::Right, KeyRepeat::Yes) && cursor % 11 < 10
            {
                cursor += 1;
            }
            if !note_focus && window.is_key_pressed(Key::Up, KeyRepeat::Yes) && cursor / 11 < 10 {
                cursor += 11;
            }
            if !note_focus && window.is_key_pressed(Key::Down, KeyRepeat::Yes) && cursor / 11 > 0 {
                cursor -= 11;
            }
        }
        let activated = if clicked {
            hover
        } else if !note_focus && window.is_key_pressed(Key::Enter, KeyRepeat::No) {
            Some(cursor)
        } else {
            None
        };
        if !rules_open
            && !referee_open
            && running.is_none()
            && replay.is_none()
            && loaded_replay.is_none()
            && (human.is_none() || human == Some(game.side))
            && let Some(s) = activated
        {
            cursor = s;
            if let Some(from) = selected {
                let m = Move { from, to: s };
                if let Ok(next) = referee.play(m) {
                    game = next;
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
        if running.is_none() && !referee.state["clock"].is_null() {
            referee.call(serde_json::json!({"command":"pause","reason":"GUI canceled search"}))?;
        }
        if replay.is_none()
            && loaded_replay.is_none()
            && game.outcome.is_none()
            && referee.state["outcome"].is_object()
        {
            game = referee.game()?;
        }
        let request = !note_focus && window.is_key_pressed(Key::Space, KeyRepeat::No)
            || panel_button == Some(2);
        if !rules_open
            && !referee_open
            && running.is_none()
            && replay.is_none()
            && loaded_replay.is_none()
            && game.outcome.is_none()
            && (request || human.is_some_and(|s| s != game.side))
        {
            match search::ready(bot, &models) {
                Err(e) => status = e,
                Ok(()) => {
                    referee.call(serde_json::json!({"command":"clock","milliseconds":325}))?;
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
        if running.is_none() && !referee.state["clock"].is_null() {
            referee.call(
                serde_json::json!({"command":"pause","reason":"GUI canceled or completed search"}),
            )?;
        }
        let mode = match human {
            Some(Side::Defender) => "Human defenders · engine attackers",
            Some(Side::Attacker) => "Human attackers · engine defenders",
            None => "Two humans · Space requests engine",
        };
        let analysis = loaded_replay
            .as_ref()
            .and_then(replay_analysis)
            .unwrap_or(&referee.state["analysis"]);
        let nodes = analysis["proof"]["certificate"]["nodes"].as_array();
        let proof_revision = analysis["revision"].as_u64();
        if !referee_open
            || nodes.is_none()
            || proof_preview
                .as_ref()
                .is_some_and(|(revision, _, _)| Some(*revision) != proof_revision)
            || (!note_focus && window.is_key_pressed(Key::Home, KeyRepeat::No))
        {
            proof_preview = None;
        }
        if referee_open
            && !note_focus
            && let Some(nodes) = nodes
            && !nodes.is_empty()
        {
            let down = window.is_key_pressed(Key::PageDown, KeyRepeat::Yes);
            let up = window.is_key_pressed(Key::PageUp, KeyRepeat::Yes);
            if down || up {
                let index = proof_preview
                    .as_ref()
                    .map(|(_, i, _)| {
                        if down {
                            (i + 1).min(nodes.len() - 1)
                        } else {
                            i.saturating_sub(1)
                        }
                    })
                    .unwrap_or(0);
                if let Some(fen) = nodes[index]["fen"].as_str()
                    && let Ok(position) = skallagrim::game::Game::from_fen(fen)
                {
                    proof_preview = Some((proof_revision.unwrap_or(0), index, position));
                }
            }
        }
        let preview_status = proof_preview.as_ref().map(|(_, index, _)| {
            format!("Fort strategy node {index} · PgUp/PgDn browse · Home live board")
        });
        let display = proof_preview
            .as_ref()
            .map(|(_, _, g)| g)
            .unwrap_or_else(|| replay.map(|r| &states[r]).unwrap_or(&game));
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
                status: preview_status.as_deref().unwrap_or(&status),
                moves: &moves[..replay.unwrap_or(moves.len())],
                thinking: running.is_some(),
                mouse,
                rules_open,
                referee: referee_open.then_some((&evidence, &note, pending, evidence_scroll)),
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
