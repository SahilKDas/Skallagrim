use skallagrim::{
    game::*,
    neural::*,
    search::{self, Bot, Limits},
};
use std::{
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};
fn position(side: Side, pieces: &[(usize, Piece)]) -> Game {
    let mut g = Game {
        board: [Piece::Empty; SQ],
        ..Game::default()
    };
    g.board[60] = Piece::King;
    for &(s, p) in pieces {
        g.board[s] = p;
    }
    g.side = side;
    g.outcome = None;
    g.history = vec![g.fen().into()];
    g
}
#[test]
fn initial_and_fen_roundtrip() {
    let g = Game::default();
    assert_eq!(
        g.board.iter().filter(|&&p| p == Piece::Attacker).count(),
        24
    );
    assert_eq!(
        g.board.iter().filter(|&&p| p == Piece::Defender).count(),
        12
    );
    assert_eq!(g.fen(), START);
    assert_eq!(g.legal_moves().len(), 116);
    for m in g.legal_moves() {
        assert_eq!(m.to_string().parse::<Move>().unwrap(), m);
    }
}
#[test]
fn rejects_invalid_input_without_mutation() {
    let mut g = Game::default();
    let fen = g.fen();
    assert!(g.play("f6a1".parse().unwrap()).is_err());
    assert_eq!(g.fen(), fen);
    for f in [
        "11 a",
        "11/11/11/11/11/11/11/11/11/11/11 a",
        "12/11/11/11/11/5k5/11/11/11/11/11 a",
    ] {
        assert!(Game::from_fen(f).is_err());
    }
    assert!("é1a2".parse::<Move>().is_err());
    assert!("a0k11".parse::<Move>().is_err());
}
#[test]
fn empty_throne_passage_and_reentry() {
    let g = position(
        Side::Attacker,
        &[(60, Piece::Empty), (61, Piece::King), (55, Piece::Attacker)],
    );
    assert!(g.legal_moves().contains(&Move { from: 55, to: 59 }));
    assert!(!g.legal_moves().contains(&Move { from: 55, to: 60 }));
    let mut g = g;
    g.side = Side::Defender;
    assert!(g.legal_moves().contains(&Move { from: 61, to: 60 }));
}
#[test]
fn sandwich_capture() {
    let mut g = position(
        Side::Attacker,
        &[
            (23, Piece::Attacker),
            (25, Piece::Defender),
            (26, Piece::Attacker),
        ],
    );
    g.play(Move { from: 23, to: 24 }).unwrap();
    assert_eq!(g.board[25], Piece::Empty);
    assert!(g.captures.contains(&25));
}
#[test]
fn moving_into_sandwich_is_safe() {
    let mut g = position(
        Side::Defender,
        &[
            (23, Piece::Attacker),
            (25, Piece::Attacker),
            (35, Piece::Defender),
        ],
    );
    g.play(Move { from: 35, to: 24 }).unwrap();
    assert_eq!(g.board[24], Piece::Defender);
}
#[test]
fn simultaneous_two_axis_capture() {
    let mut g = position(
        Side::Attacker,
        &[
            (13, Piece::Attacker),
            (25, Piece::Defender),
            (26, Piece::Attacker),
            (35, Piece::Defender),
            (46, Piece::Attacker),
        ],
    );
    g.play(Move { from: 13, to: 24 }).unwrap();
    assert_eq!(g.captures, vec![25, 35]);
}
#[test]
fn corner_is_anvil() {
    let mut g = position(
        Side::Attacker,
        &[(1, Piece::Defender), (24, Piece::Attacker)],
    );
    g.play(Move { from: 24, to: 2 }).unwrap();
    assert_eq!(g.board[1], Piece::Empty);
}
#[test]
fn occupied_throne_hostile_to_attacker_only() {
    let mut g = position(
        Side::Defender,
        &[(59, Piece::Attacker), (47, Piece::Defender)],
    );
    g.play(Move { from: 47, to: 58 }).unwrap();
    assert_eq!(g.board[59], Piece::Empty);
    let mut g = position(
        Side::Attacker,
        &[(59, Piece::Defender), (47, Piece::Attacker)],
    );
    g.play(Move { from: 47, to: 58 }).unwrap();
    assert_eq!(g.board[59], Piece::Defender);
}
#[test]
fn empty_throne_hostile_to_defender() {
    let mut g = position(
        Side::Attacker,
        &[
            (60, Piece::Empty),
            (71, Piece::King),
            (59, Piece::Defender),
            (47, Piece::Attacker),
        ],
    );
    g.play(Move { from: 47, to: 58 }).unwrap();
    assert_eq!(g.board[59], Piece::Empty);
}
#[test]
fn edge_is_not_anvil() {
    let mut g = position(
        Side::Attacker,
        &[(22, Piece::Defender), (24, Piece::Attacker)],
    );
    g.play(Move { from: 24, to: 23 }).unwrap();
    assert_eq!(g.board[22], Piece::Defender);
}
#[test]
fn shieldwall_and_king_immunity() {
    let mut g = position(
        Side::Attacker,
        &[
            (60, Piece::Empty),
            (2, Piece::King),
            (1, Piece::Defender),
            (3, Piece::Defender),
            (12, Piece::Attacker),
            (13, Piece::Attacker),
            (14, Piece::Attacker),
            (26, Piece::Attacker),
        ],
    );
    g.play(Move { from: 26, to: 4 }).unwrap();
    assert_eq!(g.board[1], Piece::Empty);
    assert_eq!(g.board[3], Piece::Empty);
    assert_eq!(g.board[2], Piece::King);
}
#[test]
fn shieldwall_requires_inward_support() {
    let mut g = position(
        Side::Attacker,
        &[
            (1, Piece::Defender),
            (2, Piece::Defender),
            (12, Piece::Attacker),
            (25, Piece::Attacker),
        ],
    );
    g.play(Move { from: 25, to: 3 }).unwrap();
    assert_eq!(g.board[1], Piece::Defender);
    assert_eq!(g.board[2], Piece::Defender);
}
#[test]
fn completing_inward_support_does_not_trigger_strict_shieldwall() {
    let mut g = position(
        Side::Attacker,
        &[
            (1, Piece::Defender),
            (2, Piece::Defender),
            (3, Piece::Attacker),
            (12, Piece::Attacker),
            (24, Piece::Attacker),
        ],
    );
    g.play(Move { from: 24, to: 13 }).unwrap();
    assert_eq!(g.board[1], Piece::Defender);
    assert_eq!(g.board[2], Piece::Defender);
}
#[test]
fn king_capture_on_throne() {
    let mut g = position(
        Side::Attacker,
        &[
            (59, Piece::Attacker),
            (61, Piece::Attacker),
            (49, Piece::Attacker),
            (73, Piece::Attacker),
        ],
    );
    g.play(Move { from: 73, to: 71 }).unwrap();
    assert_eq!(g.outcome.unwrap().reason, "king-captured");
}
#[test]
fn king_capture_adjacent_throne() {
    let mut g = position(
        Side::Attacker,
        &[
            (60, Piece::Empty),
            (59, Piece::King),
            (48, Piece::Attacker),
            (70, Piece::Attacker),
            (56, Piece::Attacker),
        ],
    );
    g.play(Move { from: 56, to: 58 }).unwrap();
    assert_eq!(g.outcome.unwrap().winner, Some(Side::Attacker));
}
#[test]
fn king_safe_on_edge() {
    let mut g = position(
        Side::Attacker,
        &[
            (60, Piece::Empty),
            (5, Piece::King),
            (4, Piece::Attacker),
            (6, Piece::Attacker),
            (27, Piece::Attacker),
        ],
    );
    g.play(Move { from: 27, to: 16 }).unwrap();
    assert_eq!(g.board[5], Piece::King);
}
#[test]
fn corner_escape() {
    let mut g = position(
        Side::Defender,
        &[
            (60, Piece::Empty),
            (5, Piece::King),
            (110, Piece::Empty),
            (33, Piece::Attacker),
        ],
    );
    g.play(Move { from: 5, to: 0 }).unwrap();
    assert_eq!(g.outcome.unwrap().reason, "corner-escape");
}
#[test]
fn encirclement_requires_all_defenders() {
    let mut g = position(Side::Attacker, &[]);
    for s in [48, 49, 50, 59, 61, 70, 71, 72] {
        g.board[s] = Piece::Attacker;
    }
    assert!(g.encircled());
    g.board[12] = Piece::Defender;
    assert!(!g.encircled());
}
#[test]
fn repetition_is_defender_loss() {
    let mut g = Game::default();
    g.history = vec![g.fen().into(); 3];
    g.adjudicate();
    assert_eq!(g.outcome.unwrap().reason, "perpetual-repetition");
}
#[test]
fn immobility_is_loss() {
    let mut g = position(Side::Attacker, &[]);
    g.adjudicate();
    assert_eq!(g.outcome.unwrap().winner, Some(Side::Defender));
}
#[test]
fn certified_exit_fort_and_breachable_wall() {
    let mut g = position(
        Side::Attacker,
        &[(60, Piece::Empty), (5, Piece::King), (2, Piece::Attacker)],
    );
    for s in [3, 7, 14, 15, 16, 17, 18] {
        g.board[s] = Piece::Defender;
    }
    assert!(g.exit_fort());
    g.board[16] = Piece::Empty;
    assert!(!g.exit_fort());
}
#[test]
fn nnue_incremental_matches_full_after_moves_and_captures() {
    let h = 8;
    let nn = Nnue {
        format: "skallagrim-nnue-v1".into(),
        hidden: h,
        weights: (0..FEATURES * h)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 300.0)
            .collect(),
        bias: vec![0.1; h],
        output: vec![0.05; h],
        output_bias: 0.0,
    };
    nn.validate().unwrap();
    let mut g = Game::default();
    let mut acc = nn.accumulator(&g);
    for i in 0..100 {
        let moves = g.legal_moves();
        if moves.is_empty() {
            break;
        }
        g.play(moves[i * 37 % moves.len()]).unwrap();
        acc = nn.update(&acc, &g);
        let fresh = nn.accumulator(&g);
        for (a, b) in acc.values.iter().zip(fresh.values) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
#[test]
fn search_respects_terminal_and_requires_models() {
    assert!(
        search::search(
            &Game::default(),
            Bot::Rho,
            &Models::default(),
            Limits::default(),
            Arc::new(AtomicBool::new(false))
        )
        .is_err()
    );
    let r = search::search(
        &Game::default(),
        Bot::Baseline,
        &Models::default(),
        Limits {
            time: Duration::from_millis(30),
            nodes: 100,
            depth: 4,
        },
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    assert!(Game::default().legal_moves().contains(&r.best.unwrap()));
    assert!(r.nodes <= 100);
}

#[test]
fn referee_draw_and_fort_requirements() {
    let mut game = Game::default();
    let original = game.fen();
    assert!(game.referee_decision("fort").is_err());
    assert!(game.outcome.is_none());
    assert_eq!(game.fen(), original);
    game.referee_decision("draw").unwrap();
    assert_eq!(game.outcome.as_ref().unwrap().winner, None);
    assert!(game.legal_moves().is_empty());
    assert!(game.referee_decision("draw").is_err());
}
#[test]
fn scarce_attackers_cannot_break_a_fort_requiring_two_capturers() {
    let mut game = position(
        Side::Attacker,
        &[(60, Piece::Empty), (5, Piece::King), (99, Piece::Attacker)],
    );
    for s in [3, 7, 14, 15, 16, 17, 18] {
        game.board[s] = Piece::Defender;
    }
    assert!(game.exit_fort());
    game.board[16] = Piece::Empty;
    assert!(!game.exit_fort());
}

#[test]
fn neural_search_handles_immediate_wins_for_both_roles() {
    let models = Models {
        nnue: Some(Nnue {
            format: "skallagrim-nnue-v1".into(),
            hidden: 1,
            weights: vec![0.0; FEATURES],
            bias: vec![0.0],
            output: vec![0.0],
            output_bias: 0.0,
        }),
        ..Models::default()
    };
    let attackers = position(
        Side::Attacker,
        &[
            (59, Piece::Attacker),
            (61, Piece::Attacker),
            (49, Piece::Attacker),
            (73, Piece::Attacker),
        ],
    );
    let defenders = position(
        Side::Defender,
        &[(60, Piece::Empty), (5, Piece::King), (33, Piece::Attacker)],
    );
    for game in [attackers, defenders] {
        for bot in [Bot::Iota, Bot::Beta] {
            let result = search::search(
                &game,
                bot,
                &models,
                Limits {
                    time: Duration::from_secs(5),
                    nodes: 1024,
                    depth: 2,
                },
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
            let mut next = game.clone();
            next.play(result.best.unwrap()).unwrap();
            assert_eq!(
                next.outcome.as_ref().and_then(|o| o.winner),
                Some(game.side),
                "{bot:?} failed an immediate {:?} win",
                game.side
            );
        }
    }
}
#[test]
fn rules_and_captures_preserve_all_board_symmetries() {
    fn transform(s: usize, symmetry: usize) -> usize {
        let (mut x, mut y) = (s % 11, s / 11);
        if symmetry >= 4 {
            x = 10 - x;
        }
        for _ in 0..symmetry % 4 {
            (x, y) = (10 - y, x);
        }
        y * 11 + x
    }
    let mut original = Game::default();
    let mut reflected: Vec<Game> = (0..8)
        .map(|symmetry| {
            let mut game = original.clone();
            for s in 0..SQ {
                game.board[transform(s, symmetry)] = original.board[s];
            }
            game.history = vec![game.fen().into()];
            game
        })
        .collect();
    for ply in 0..80 {
        let legal = original.legal_moves();
        if legal.is_empty() {
            break;
        }
        let mv = legal[(ply * 37 + 11) % legal.len()];
        original.play(mv).unwrap();
        for (symmetry, game) in reflected.iter_mut().enumerate() {
            game.play(Move {
                from: transform(mv.from, symmetry),
                to: transform(mv.to, symmetry),
            })
            .unwrap();
            for s in 0..SQ {
                assert_eq!(game.board[transform(s, symmetry)], original.board[s]);
            }
            assert_eq!(game.outcome, original.outcome);
            let mut captured: Vec<_> = original
                .captures
                .iter()
                .map(|&s| transform(s, symmetry))
                .collect();
            captured.sort_unstable();
            assert_eq!(game.captures, captured);
        }
    }
}
