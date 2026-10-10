use serde_json::json;
use skallagrim::{game::Move, referee::Referee, replay::Replay};
use std::time::{Duration, Instant};

#[test]
fn rejects_illegal_and_stale_moves_without_mutating_board() {
    let mut r = Referee::default();
    let now = Instant::now();
    let before = r.game.fen();
    assert!(
        r.command(&json!({"command":"play","move":"a1a2"}), now)
            .is_err()
    );
    assert_eq!(r.game.fen(), before);
    let mv = r.game.legal_moves()[0];
    assert!(
        r.command(
            &json!({"command":"play","move":mv.to_string(),"revision":99}),
            now
        )
        .is_err()
    );
    assert_eq!(r.game.fen(), before);
}
#[test]
fn expired_clock_forfeits_without_accepting_late_move() {
    let mut r = Referee::default();
    let now = Instant::now();
    let mv = r.game.legal_moves()[0];
    let before = r.game.fen();
    r.command(&json!({"command":"clock","milliseconds":300}), now)
        .unwrap();
    assert!(
        r.command(
            &json!({"command":"play","move":mv.to_string()}),
            now + Duration::from_millis(300)
        )
        .is_err()
    );
    assert_eq!(r.game.fen(), before);
    assert_eq!(r.game.outcome.as_ref().unwrap().reason, "timeout");
    Replay::from_value(&r.export()).unwrap();
}
#[test]
fn timely_move_and_undo_have_replayable_audit_trail() {
    let mut r = Referee::default();
    let now = Instant::now();
    r.command(&json!({"command":"new"}), now).unwrap();
    let mv = r.game.legal_moves()[0];
    r.command(&json!({"command":"clock","milliseconds":300}), now)
        .unwrap();
    r.command(
        &json!({"command":"play","move":mv.to_string()}),
        now + Duration::from_millis(299),
    )
    .unwrap();
    let mv = r.game.legal_moves()[0];
    r.command(
        &json!({"command":"play","move":mv.to_string()}),
        now + Duration::from_millis(300),
    )
    .unwrap();
    r.command(
        &json!({"command":"undo","count":1}),
        now + Duration::from_millis(301),
    )
    .unwrap();
    Replay::from_value(&r.export()).unwrap();
}
#[test]
fn human_ruling_requires_reason_and_identity_and_tamper_is_rejected() {
    let mut r = Referee::default();
    let now = Instant::now();
    assert!(
        r.command(&json!({"command":"adjudicate","claim":"draw"}), now)
            .is_err()
    );
    assert!(r.game.outcome.is_none());
    r.command(&json!({"command":"adjudicate","claim":"draw","operator":"Referee Alice","reason":"Both players agree the unresolved position cannot finish"}),now).unwrap();
    let export = r.export();
    Replay::from_value(&export).unwrap();
    let mut corrupted = export.clone();
    corrupted["referee"]["events"][0]["evidence"]["reason"] = json!("");
    assert!(Replay::from_value(&corrupted).is_err());
    let mut corrupted = export;
    corrupted["referee"]["events"][0]["fen"] = json!("bad");
    assert!(Replay::from_value(&corrupted).is_err());
}
#[test]
fn unknown_and_stale_analysis_do_not_adjudicate() {
    let mut r = Referee::default();
    let now = Instant::now();
    r.evidence(
        0,
        json!({"revision":0,"claim":"fort","status":"unknown","reason":"outside static certificate"}),
    );
    assert!(r.game.outcome.is_none());
    let mv: Move = r.game.legal_moves()[0];
    r.command(&json!({"command":"play","move":mv.to_string()}), now)
        .unwrap();
    r.evidence(0, json!({"status":"proven"}));
    assert!(r.snapshot()["analysis"].is_null());
    Replay::from_value(&r.export()).unwrap();
}
#[test]
fn invalid_new_position_is_atomic_and_clock_cannot_be_rearmed() {
    let mut r = Referee::default();
    let now = Instant::now();
    let before = r.game.fen();
    assert!(
        r.command(&json!({"command":"new","moves":["bad"]}), now)
            .is_err()
    );
    assert_eq!(r.game.fen(), before);
    r.command(&json!({"command":"clock","milliseconds":20}), now)
        .unwrap();
    assert!(
        r.command(&json!({"command":"clock","milliseconds":1000}), now)
            .is_err()
    );
    r.tick(now + Duration::from_millis(21));
    assert_eq!(r.game.outcome.unwrap().reason, "timeout");
}
#[test]
fn opening_and_administrative_result_are_replayable() {
    let mut r = Referee::default();
    let now = Instant::now();
    let mv = r.game.legal_moves()[0];
    r.command(&json!({"command":"new","moves":[mv.to_string()]}), now)
        .unwrap();
    r.command(
        &json!({"command":"forfeit","reason":"engine-failure","evidence":"broken pipe"}),
        now,
    )
    .unwrap();
    Replay::from_value(&r.export()).unwrap();
}
#[test]
fn authoritative_repetition_and_terminal_rules_have_audited_rulings() {
    use skallagrim::game::{Game, Piece, SQ, Side};
    let now = Instant::now();
    let mut r = Referee::default();
    for _ in 0..2 {
        for text in ["e1e2", "f4e4", "e2e1", "e4f4"] {
            r.command(&json!({"command":"play","move":text}), now)
                .unwrap();
        }
    }
    assert_eq!(
        r.game.outcome.as_ref().unwrap().reason,
        "perpetual-repetition"
    );
    Replay::from_value(&r.export()).unwrap();
    let mut fixture = Game {
        board: [Piece::Empty; SQ],
        side: Side::Attacker,
        ..Game::default()
    };
    fixture.board[60] = Piece::King;
    for sq in [59, 61, 49, 73] {
        fixture.board[sq] = Piece::Attacker;
    }
    r.command(&json!({"command":"new","fen":fixture.fen()}), now)
        .unwrap();
    r.command(&json!({"command":"play","move":"h7f7"}), now)
        .unwrap();
    assert_eq!(r.game.outcome.as_ref().unwrap().reason, "king-captured");
    assert!(r.game.king().is_none());
    Replay::from_value(&r.export()).unwrap();
    fixture.board = [Piece::Empty; SQ];
    fixture.board[5] = Piece::King;
    fixture.board[33] = Piece::Attacker;
    fixture.side = Side::Defender;
    r.command(&json!({"command":"new","fen":fixture.fen()}), now)
        .unwrap();
    r.command(&json!({"command":"play","move":"f1a1"}), now)
        .unwrap();
    assert_eq!(r.game.outcome.as_ref().unwrap().reason, "corner-escape");
    Replay::from_value(&r.export()).unwrap();
    fixture.board = [Piece::Empty; SQ];
    fixture.board[60] = Piece::King;
    fixture.side = Side::Attacker;
    r.command(&json!({"command":"new","fen":fixture.fen()}), now)
        .unwrap();
    assert_eq!(r.game.outcome.as_ref().unwrap().reason, "immobility");
    Replay::from_value(&r.export()).unwrap();
    for sq in [48, 49, 50, 59, 61, 70, 71, 72] {
        fixture.board[sq] = Piece::Attacker;
    }
    r.command(&json!({"command":"new","fen":fixture.fen()}), now)
        .unwrap();
    assert_eq!(r.game.outcome.as_ref().unwrap().reason, "encirclement");
    Replay::from_value(&r.export()).unwrap();
    fixture.board = [Piece::Empty; SQ];
    fixture.board[5] = Piece::King;
    fixture.board[2] = Piece::Attacker;
    for sq in [3, 7, 14, 15, 16, 17, 18] {
        fixture.board[sq] = Piece::Defender;
    }
    r.command(&json!({"command":"new","fen":fixture.fen()}), now)
        .unwrap();
    assert_eq!(r.game.outcome.as_ref().unwrap().reason, "exit-fort");
    Replay::from_value(&r.export()).unwrap();
}
#[test]
fn malformed_controls_do_not_reset_or_relabel_the_game() {
    let now = Instant::now();
    let mut r = Referee::default();
    let before = r.game.fen();
    for request in [
        json!({"command":"new","fen":0}),
        json!({"command":"new","moves":"bad"}),
        json!({"command":"new","revision":"bad"}),
        json!({"command":"forfeit","reason":"move-limit-adjudication"}),
    ] {
        assert!(r.command(&request, now).is_err());
        assert_eq!(r.game.fen(), before);
        assert!(r.game.outcome.is_none());
    }
    r.command(&json!({"command":"clock","milliseconds":300}), now)
        .unwrap();
    assert!(r.command(&json!({"command":"pause"}), now).is_err());
    assert!(r.snapshot()["clock"].is_object());
}
#[test]
fn referee_confirm_and_close_controls_remain_visible_when_resized() {
    use skallagrim::render::{Layout, Renderer, View, referee_scale};
    let game = skallagrim::game::Game::default();
    let renderer = Renderer::default();
    let evidence = vec![
        "draw: unknown".to_string(),
        "Budget exhausted; human decision required".to_string(),
    ];
    for (w, h) in [(1180, 880), (900, 600)] {
        let pix = renderer.draw(
            w,
            h,
            View {
                game: &game,
                selected: None,
                hover: None,
                cursor: 60,
                bot: "Iota",
                mode: "Human defenders",
                status: "Referee review",
                moves: &[],
                thinking: false,
                mouse: None,
                rules_open: false,
                referee: Some((&evidence, "Human reasoning", Some("draw"), 0)),
            },
        );
        let layout = Layout::new(w, h);
        for logical_y in [659.0, 757.0] {
            let y = (98.0 + (logical_y - 98.0) * referee_scale(h)) as u32;
            assert!(y < h - 42);
            let color = pix.pixel((layout.panel + 18.0) as u32, y).unwrap();
            assert_eq!((color.red(), color.green(), color.blue()), (43, 57, 72));
        }
        if h == 600 && std::env::var_os("SKALLAGRIM_RENDER_TEST").is_some() {
            std::fs::create_dir_all("runs").unwrap();
            pix.save_png("runs/referee-panel-small.png").unwrap();
        }
    }
}
