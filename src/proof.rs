//! Bounded, exact reachability certificate for the qualitative draw rule.
//! A budget exhaustion is unknown, never a draw. This is not a minimax claim.
use crate::game::{Game, History, Move};
use serde::Serialize;
use std::{
    collections::HashSet,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

#[derive(Debug, Serialize, PartialEq, Eq)]
pub enum ProofStatus {
    ProvenDraw,
    FinishReachable,
    Unknown,
}
#[derive(Debug, Serialize)]
pub struct DrawProof {
    pub status: ProofStatus,
    pub positions: usize,
    pub reason: String,
    /// A cooperative legal continuation to a decisive outcome, when found.
    pub witness: Vec<String>,
}

pub fn prove_draw(
    game: &Game,
    max_positions: usize,
    time: Duration,
    stop: &AtomicBool,
) -> DrawProof {
    let start = Instant::now();
    let result = |status, positions, reason: &str, witness| DrawProof {
        status,
        positions,
        reason: reason.into(),
        witness,
    };
    if game.outcome.is_some() {
        return result(ProofStatus::FinishReachable, 1, "already terminal", vec![]);
    }
    let mut root = game.clone();
    root.history = History::default();
    root.history.push(root.fen().into());
    root.adjudicate();
    if let Some(outcome) = &root.outcome {
        return result(ProofStatus::FinishReachable, 1, &outcome.reason, vec![]);
    }
    let mut seen = HashSet::from([root.fen()]);
    let mut states: Vec<(Game, Option<(usize, Move)>)> = vec![(root, None)];
    let mut cursor = 0;
    while cursor < states.len() {
        if start.elapsed() >= time || stop.load(Ordering::Relaxed) {
            return result(
                ProofStatus::Unknown,
                states.len(),
                "time limit or cancellation",
                vec![],
            );
        }
        for mv in states[cursor].0.legal_moves() {
            if start.elapsed() >= time || stop.load(Ordering::Relaxed) {
                return result(
                    ProofStatus::Unknown,
                    states.len(),
                    "time limit or cancellation",
                    vec![],
                );
            }
            let mut child = states[cursor].0.clone();
            // Repetition is deliberately excluded: rule 10 asks whether play
            // can finish, rather than whether a repeated sequence incurs a
            // penalty. Captures, escapes, forts, encirclement and immobility
            // remain active. Board + side is then a complete transition key.
            child.history = History::default();
            child.history.push(child.fen().into());
            child.play_unchecked(mv);
            if let Some(outcome) = &child.outcome {
                let mut witness = vec![mv.to_string()];
                let mut at = cursor;
                while let Some((parent, previous)) = states[at].1 {
                    witness.push(previous.to_string());
                    at = parent;
                }
                witness.reverse();
                return result(
                    ProofStatus::FinishReachable,
                    states.len(),
                    &outcome.reason,
                    witness,
                );
            }
            let key = child.fen();
            if !seen.contains(&key) {
                if states.len() >= max_positions {
                    return result(ProofStatus::Unknown, states.len(), "position limit", vec![]);
                }
                seen.insert(key);
                states.push((child, Some((cursor, mv))));
            }
        }
        cursor += 1;
    }
    result(
        ProofStatus::ProvenDraw,
        states.len(),
        "closed reachable graph contains no decisive outcome",
        vec![],
    )
}
