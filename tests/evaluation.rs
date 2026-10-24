use skallagrim::{
    evaluation,
    game::{Game, Outcome, Piece, SQ, Side},
    neural::{Cnn, Conv, FEATURES, Models, Nnue},
    search::{self, Bot, Limits},
};
use std::{
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

fn transform(game: &Game, rotation: usize, reflect: bool) -> Game {
    let mut next = game.clone();
    next.board = [Piece::Empty; SQ];
    for s in 0..SQ {
        let (mut x, mut y) = (s % 11, s / 11);
        for _ in 0..rotation {
            (x, y) = (10 - y, x);
        }
        if reflect {
            x = 10 - x;
        }
        next.board[y * 11 + x] = game.board[s];
    }
    next
}

#[test]
fn symmetric_finite_bounded_and_terminal_scores() {
    let mut game = Game::default();
    for ply in 0..60 {
        if game.outcome.is_some() {
            break;
        }
        let score = evaluation::evaluate(&game);
        assert!(score.is_finite() && score.abs() <= 0.95);
        for r in 0..4 {
            for f in [false, true] {
                assert!((score - evaluation::evaluate(&transform(&game, r, f))).abs() < 1e-6);
            }
        }
        let legal = game.legal_moves();
        game.play(legal[(ply * 31 + 3) % legal.len()]).unwrap();
    }
    for (winner, expected) in [
        (Some(Side::Attacker), 1.0),
        (Some(Side::Defender), -1.0),
        (None, 0.0),
    ] {
        game.outcome = Some(Outcome {
            winner,
            reason: "test".into(),
        });
        for weight in [0, 25, 100] {
            assert_eq!(evaluation::blend(&game, -0.7, weight), expected);
        }
    }
}

#[test]
fn clear_corner_route_favors_defenders_and_blend_preserves_control() {
    let mut game = Game {
        board: [Piece::Empty; SQ],
        ..Game::default()
    };
    game.board[5] = Piece::King;
    game.board[35] = Piece::Attacker;
    game.board[2] = Piece::Attacker;
    let blocked = evaluation::evaluate(&game);
    game.board[2] = Piece::Empty;
    game.board[36] = Piece::Attacker;
    assert!(evaluation::evaluate(&game) < blocked);
    assert_eq!(evaluation::blend(&game, 0.375, 0), 0.375);
    assert_eq!(
        evaluation::blend(&game, 0.375, 100),
        evaluation::evaluate(&game)
    );
    assert!(evaluation::blend(&game, f32::NAN, 25).is_finite());
}

#[test]
fn all_three_searches_use_the_configured_value_blend() {
    let conv = |input, output, kernel| Conv {
        input,
        output,
        kernel,
        weights: vec![0.0; input * output * kernel * kernel],
        bias: vec![0.0; output],
    };
    let mut models = Models {
        nnue: Some(Nnue {
            format: "skallagrim-nnue-v1".into(),
            hidden: 1,
            weights: vec![0.0; FEATURES],
            bias: vec![0.0],
            output: vec![0.0],
            output_bias: 0.5,
        }),
        cnn: Some(Cnn {
            format: "skallagrim-cnn-v1".into(),
            channels: 1,
            stem: conv(6, 1, 3),
            blocks: vec![],
            policy: conv(1, 40, 1),
            value: vec![0.0],
            value_bias: 0.5,
        }),
        handcrafted_weight: Some(0),
    };
    let game = Game::default();
    for bot in [Bot::Iota, Bot::Beta, Bot::Rho] {
        for weight in [0, 25, 100] {
            models.handcrafted_weight = Some(weight);
            let result = search::search(
                &game,
                bot,
                &models,
                Limits {
                    nodes: if bot == Bot::Rho { 1 } else { 0 },
                    time: Duration::from_secs(1),
                    depth: 2,
                },
                Arc::new(AtomicBool::new(false)),
            )
            .unwrap();
            assert!((result.value - evaluation::blend(&game, 0.5f32.tanh(), weight)).abs() < 1e-6);
        }
    }
}
