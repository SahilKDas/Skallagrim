use serde_json::{Value, json};
use skallagrim::{
    proof::{ProofStatus, prove_draw},
    referee::Referee,
};
use std::{
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
fn persist(
    referee: &Referee,
    directory: &std::path::Path,
    session: &str,
    game: u64,
    signature: &mut Option<(u64, u64)>,
) -> Result<(), String> {
    let replay = referee.export();
    let current = (
        referee.revision,
        replay["referee"]["events"].as_array().unwrap().len() as u64,
    );
    if *signature == Some(current) {
        return Ok(());
    }
    let path = directory.join(format!("{session}-game-{game:06}.json"));
    let temporary = path.with_extension("json.tmp");
    std::fs::write(
        &temporary,
        serde_json::to_vec_pretty(&replay).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    std::fs::rename(&temporary, &path).map_err(|e| e.to_string())?;
    *signature = Some(current);
    Ok(())
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let directory = args
        .iter()
        .position(|a| a == "--log-dir")
        .and_then(|i| args.get(i + 1))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("runs/referee"));
    if let Err(e) = std::fs::create_dir_all(&directory) {
        eprintln!("Referee cannot create replay directory: {e}");
        return;
    }
    let session = format!(
        "referee-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    );
    let mut game_number = 0;
    let mut signature = None;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            match line {
                Ok(line) => {
                    if tx.send((Instant::now(), line)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    let (proof_tx, proof_rx) = mpsc::channel();
    let mut referee = Referee::default();
    let mut analyzing = false;
    let mut active: Option<(u64, Arc<AtomicBool>)> = None;
    loop {
        if let Some((revision, flag)) = &active
            && (*revision != referee.revision || referee.game.outcome.is_some())
        {
            flag.store(true, Ordering::Relaxed);
        }
        if let Err(e) = persist(&referee, &directory, &session, game_number, &mut signature) {
            eprintln!("Referee cannot persist ruling: {e}");
            break;
        }
        match rx.recv_timeout(Duration::from_millis(5)) {
            Ok((arrival, line)) => {
                let request: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(e) => {
                        println!("{}", json!({"ok":false,"error":e.to_string()}));
                        continue;
                    }
                };
                if request["command"] == "quit" {
                    break;
                }
                referee.tick(arrival);
                let result = if request["command"] == "analyze" {
                    if !["draw", "fort"].contains(&request["claim"].as_str().unwrap_or("draw")) {
                        Err("invalid analysis claim".into())
                    } else if analyzing {
                        Err("analysis already running".to_string())
                    } else if request["revision"]
                        .as_u64()
                        .is_some_and(|r| r != referee.revision)
                    {
                        Err("stale position revision".into())
                    } else {
                        let revision = referee.revision;
                        let game = referee.game.clone();
                        let claim = request["claim"].as_str().unwrap_or("draw").to_string();
                        let nodes =
                            request["nodes"].as_u64().unwrap_or(10000).clamp(1, 100000) as usize;
                        let ms = request["milliseconds"]
                            .as_u64()
                            .unwrap_or(500)
                            .clamp(1, 10000);
                        let tx = proof_tx.clone();
                        let stop = Arc::new(AtomicBool::new(false));
                        active = Some((revision, stop.clone()));
                        analyzing = true;
                        std::thread::spawn(move || {
                            let began = Instant::now();
                            let mut verified = None;
                            let result = if claim == "fort" {
                                if game.exit_fort() {
                                    json!({"claim":"fort","status":"proven","reason":"Conservative static exit-fort certificate","revision":revision})
                                } else {
                                    let proof = skallagrim::fort::prove(
                                        &game,
                                        nodes,
                                        Duration::from_millis(ms),
                                        &stop,
                                    );
                                    if let Some(cert) = &proof.certificate {
                                        verified = skallagrim::fort::verify_bounded(
                                            &game,
                                            cert,
                                            Duration::from_millis(ms)
                                                .saturating_sub(began.elapsed()),
                                            &stop,
                                        )
                                        .ok();
                                    }
                                    if proof.status == "proven" && verified.is_none() {
                                        json!({"claim":"fort","status":"unknown","reason":"Certificate verification budget exhausted or cancelled","revision":revision})
                                    } else {
                                        json!({"claim":"fort","status":proof.status,"reason":proof.reason,"proof":proof,"revision":revision})
                                    }
                                }
                            } else {
                                let proof =
                                    prove_draw(&game, nodes, Duration::from_millis(ms), &stop);
                                json!({"claim":"draw","status":match proof.status {ProofStatus::ProvenDraw=>"proven",ProofStatus::FinishReachable=>"finish-reachable",ProofStatus::Unknown=>"unknown"},"proof":proof,"revision":revision})
                            };
                            let _ = tx.send((revision, result, verified));
                        });
                        Ok(json!({"status":"analyzing","revision":revision}))
                    }
                } else {
                    referee.command(&request, arrival)
                };
                if request["command"] == "new" && result.is_ok() {
                    game_number += 1;
                }
                if let Err(e) = persist(&referee, &directory, &session, game_number, &mut signature)
                {
                    println!(
                        "{}",
                        json!({"id":request["id"],"ok":false,"error":format!("Cannot persist referee replay: {e}"),"state":referee.snapshot()})
                    );
                    break;
                }
                println!(
                    "{}",
                    match result {
                        Ok(result) =>
                            json!({"id":request["id"],"ok":true,"result":result,"state":referee.snapshot()}),
                        Err(error) =>
                            json!({"id":request["id"],"ok":false,"error":error,"state":referee.snapshot()}),
                    }
                );
                let _ = io::stdout().flush();
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                referee.tick(Instant::now());
                while let Ok((revision, result, verified)) = proof_rx.try_recv() {
                    analyzing = false;
                    active = None;
                    referee.tick(Instant::now());
                    referee.evidence(revision, result);
                    if let Some(proof) = verified {
                        referee.verified_fort(revision, proof);
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    if let Some((_, stop)) = active {
        stop.store(true, Ordering::Relaxed);
    }
}
