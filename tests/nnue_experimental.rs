use skallagrim::{
    game::Game,
    neural::{EXPERIMENTAL_FEATURES, FEATURES, Nnue, experimental_features},
};

#[test]
fn experimental_incremental_matches_rebuild() {
    let n = Nnue {
        format: "skallagrim-nnue-v2".into(),
        hidden: 8,
        weights: (0..EXPERIMENTAL_FEATURES * 8)
            .map(|i| (i % 17) as f32 / 1000.0)
            .collect(),
        bias: vec![0.0; 8],
        output: vec![0.1; 8],
        output_bias: 0.0,
    };
    n.validate().unwrap();
    let mut game = Game::default();
    let mut accumulator = n.accumulator(&game);
    for i in 0..40 {
        let active = experimental_features(&game);
        assert_eq!(active.iter().filter(|&&f| f >= FEATURES).count(), 4);
        assert!(active.iter().all(|&f| f < EXPERIMENTAL_FEATURES));
        if game.outcome.is_some() {
            break;
        }
        let legal = game.legal_moves();
        game.play(legal[i % legal.len()]).unwrap();
        if game.outcome.is_some() {
            break;
        }
        accumulator = n.update(&accumulator, &game);
        let rebuilt = n.accumulator(&game);
        for (a, b) in accumulator.values.iter().zip(&rebuilt.values) {
            assert!((a - b).abs() < 2e-5);
        }
        assert!((n.value(&accumulator) - n.value(&rebuilt)).abs() < 2e-5);
    }
}

#[test]
fn experimental_format_rejects_old_shape() {
    let n = Nnue {
        format: "skallagrim-nnue-v2".into(),
        hidden: 1,
        weights: vec![0.0; FEATURES],
        bias: vec![0.0],
        output: vec![0.0],
        output_bias: 0.0,
    };
    assert!(n.validate().is_err());
}
