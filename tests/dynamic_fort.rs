use serde_json::json;
use skallagrim::{fort, game::*, referee::Referee, replay::Replay};
use std::{
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};
fn dynamic(side: Side) -> Game {
    let mut game = Game {
        board: [Piece::Empty; SQ],
        side,
        outcome: None,
        ..Game::default()
    };
    game.board[1] = Piece::King;
    for s in [2, 11, 12] {
        game.board[s] = Piece::Defender;
    }
    for s in [3, 22, 24] {
        game.board[s] = Piece::Attacker;
    }
    game.history = vec![game.fen().into()].into();
    game.adjudicate();
    game
}
#[test]
fn dynamic_strategy_verifies_and_covers_attackers() {
    for side in [Side::Attacker, Side::Defender] {
        let game = dynamic(side);
        assert!(!game.exit_fort());
        assert!(game.outcome.is_none());
        let proof = fort::prove(
            &game,
            100000,
            Duration::from_secs(10),
            &AtomicBool::new(false),
        );
        assert_eq!(
            proof.status, "proven",
            "{}; {} positions",
            proof.reason, proof.positions
        );
        let cert = proof.certificate.unwrap();
        fort::verify(&game, &cert).unwrap();
        if side == Side::Attacker {
            assert_eq!(cert.nodes[0].edges.len(), game.legal_moves().len());
        }
        assert!(cert.nodes.iter().skip(1).any(|n| !n.edges.is_empty()) || side == Side::Defender);
    }
}
#[test]
fn rejects_missing_response_forged_successor_and_region() {
    let game = dynamic(Side::Attacker);
    let cert = fort::prove(
        &game,
        100000,
        Duration::from_secs(10),
        &AtomicBool::new(false),
    )
    .certificate
    .unwrap();
    let mut bad = cert.clone();
    bad.nodes[0].edges.pop();
    assert!(fort::verify(&game, &bad).is_err());
    let mut bad = cert.clone();
    bad.nodes[0].edges[0].next = usize::MAX;
    assert!(fort::verify(&game, &bad).is_err());
    let mut bad = cert.clone();
    bad.region.push(SQ);
    assert!(fort::verify(&game, &bad).is_err());
    let mut bad = cert;
    bad.nodes[0].edges[0].mv = "b1b2".into();
    assert!(fort::verify(&game, &bad).is_err());
}
#[test]
fn budget_and_cancel_are_unknown() {
    let game = dynamic(Side::Attacker);
    for (limit, time, cancel) in [
        (1, Duration::from_secs(1), false),
        (100000, Duration::ZERO, false),
        (100000, Duration::from_secs(1), true),
    ] {
        let proof = fort::prove(&game, limit, time, &AtomicBool::new(cancel));
        assert_eq!(proof.status, "unknown");
        assert!(proof.certificate.is_none());
    }
    assert_eq!(
        fort::prove(
            &Game::default(),
            1000,
            Duration::from_secs(1),
            &AtomicBool::new(false)
        )
        .status,
        "unknown"
    );
}
#[test]
fn mobility_breach_and_terminal_repetition_are_not_proven() {
    let mut blocked = dynamic(Side::Defender);
    blocked.board = [Piece::Empty; SQ];
    blocked.board[5] = Piece::King;
    for s in [4, 6, 16] {
        blocked.board[s] = Piece::Defender;
    }
    blocked.board[22] = Piece::Attacker;
    blocked.outcome = None;
    assert_eq!(
        fort::prove(
            &blocked,
            1000,
            Duration::from_secs(1),
            &AtomicBool::new(false)
        )
        .status,
        "unknown"
    );
    let mut game = dynamic(Side::Attacker);
    game.board[0] = Piece::Empty;
    game.board[12] = Piece::Empty;
    assert_eq!(
        fort::prove(
            &game,
            10000,
            Duration::from_secs(1),
            &AtomicBool::new(false)
        )
        .status,
        "unknown"
    );
    let mut game = dynamic(Side::Attacker);
    game.board[0] = Piece::Empty;
    game.board[2] = Piece::Empty;
    assert_eq!(
        fort::prove(
            &game,
            10000,
            Duration::from_secs(1),
            &AtomicBool::new(false)
        )
        .status,
        "unknown"
    );
    let mut game = dynamic(Side::Attacker);
    game.history = vec![game.fen().into(); 3].into();
    game.adjudicate();
    assert_eq!(
        game.outcome.as_ref().unwrap().reason,
        "perpetual-repetition"
    );
    assert_eq!(
        fort::prove(&game, 1000, Duration::from_secs(1), &AtomicBool::new(false)).status,
        "unknown"
    );
}
#[test]
fn shieldwall_moves_are_included_in_universal_coverage() {
    let mut game = dynamic(Side::Attacker);
    for s in [8, 9] {
        game.board[s] = Piece::Defender;
    }
    for s in [19, 20, 40] {
        game.board[s] = Piece::Attacker;
    }
    game.history = vec![game.fen().into()].into();
    let mut child = game.clone();
    child.play("h4h1".parse().unwrap()).unwrap();
    assert!(child.captures.contains(&8) && child.captures.contains(&9));
    let proof = fort::prove(
        &game,
        100000,
        Duration::from_secs(10),
        &AtomicBool::new(false),
    );
    assert_eq!(proof.status, "proven", "{}", proof.reason);
    let cert = proof.certificate.unwrap();
    fort::verify(&game, &cert).unwrap();
    assert!(cert.nodes[0].edges.iter().any(|e| e.mv == "h4h1"));
}
#[test]
fn automatic_ruling_audits_and_stale_proof_is_dropped() {
    let game = dynamic(Side::Defender);
    let proof = fort::prove(
        &game,
        100000,
        Duration::from_secs(10),
        &AtomicBool::new(false),
    );
    let cert = proof.certificate.as_ref().unwrap().clone();
    let verified = fort::verify(&game, &cert).unwrap();
    let mut referee = Referee::default();
    referee
        .command(&json!({"command":"new","fen":game.fen()}), Instant::now())
        .unwrap();
    let revision = referee.revision;
    referee.evidence(
        revision,
        json!({"claim":"fort","status":"proven","proof":proof,"revision":revision}),
    );
    referee.verified_fort(revision, verified);
    assert_eq!(
        referee.game.outcome.as_ref().unwrap().reason,
        "exit-fort-dynamic"
    );
    Replay::from_value(&referee.export()).unwrap();
    let mut forged = referee.export();
    forged["referee"]["events"][1]["evidence"]["proof"]["certificate"]["region"] = json!([120]);
    assert!(Replay::from_value(&forged).is_err());
    referee
        .command(&json!({"command":"new"}), Instant::now())
        .unwrap();
    referee.verified_fort(revision, fort::verify(&game, &cert).unwrap());
    assert!(referee.game.outcome.is_none());
}
