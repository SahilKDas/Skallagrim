//! Conservative dynamic exit-fort certificates. Repetition is deliberately
//! excluded from the invariant graph: a verified fort is awarded immediately,
//! not played out as a perpetual sequence. All other terminal rules remain on.
use crate::game::{Game, History, Move, Piece, SQ, Side, edge, step};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArcMove {
    pub mv: String,
    pub next: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FortNode {
    pub fen: String,
    pub edges: Vec<ArcMove>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Certificate {
    pub version: u32,
    pub root: String,
    pub region: Vec<usize>,
    pub nodes: Vec<FortNode>,
}
#[derive(Debug, Serialize)]
pub struct FortProof {
    pub status: &'static str,
    pub positions: usize,
    pub reason: String,
    pub certificate: Option<Certificate>,
}
/// Constructed only by the certificate verifier; cannot be forged by RPC.
pub struct VerifiedFort {
    root: String,
}
impl VerifiedFort {
    pub fn matches(&self, game: &Game) -> bool {
        self.root == game.fen()
    }
}

fn normalized(mut game: Game) -> Game {
    game.history = History::default();
    game.history.push(game.fen().into());
    game
}
fn transition(game: &Game, mv: Move) -> Game {
    let mut next = normalized(game.clone());
    next.play_unchecked(mv);
    next
}
fn region(game: &Game) -> Option<Vec<usize>> {
    let king = game.king()?;
    if !edge(king) {
        return None;
    }
    let mut seen = HashSet::from([king]);
    let mut queue = VecDeque::from([king]);
    while let Some(s) = queue.pop_front() {
        if game.board[s] == Piece::Attacker {
            return None;
        }
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            if let Some(t) = step(s, dx, dy)
                && game.board[t] != Piece::Defender
                && seen.insert(t)
            {
                queue.push_back(t);
            }
        }
    }
    let mut cells: Vec<_> = seen.into_iter().collect();
    cells.sort_unstable();
    Some(cells)
}
fn safe(game: &Game, cells: &[usize]) -> bool {
    if let Some(outcome) = &game.outcome {
        return outcome.winner == Some(Side::Defender);
    }
    game.king().is_some_and(|k| {
        edge(k)
            && cells.contains(&k)
            && game
                .moves_for(Side::Defender)
                .iter()
                .any(|m| m.from == k && cells.contains(&m.to))
    }) && cells.iter().all(|&s| game.board[s] != Piece::Attacker)
}

/// Verify legal transitions and an inductive closed strategy, without search.
pub fn verify(game: &Game, cert: &Certificate) -> Result<VerifiedFort, String> {
    verify_bounded(game, cert, Duration::from_secs(10), &AtomicBool::new(false))
}
pub fn verify_bounded(
    game: &Game,
    cert: &Certificate,
    time: Duration,
    stop: &AtomicBool,
) -> Result<VerifiedFort, String> {
    let started = Instant::now();
    let check = || {
        if started.elapsed() >= time || stop.load(Ordering::Relaxed) {
            Err("Fort verification time limit or cancellation".to_string())
        } else {
            Ok(())
        }
    };
    check()?;
    if cert.version != 1
        || cert.root != game.fen()
        || cert.nodes.is_empty()
        || cert.nodes.len() > 100000
        || cert.region.is_empty()
        || cert.region.len() > SQ
        || cert.region.iter().any(|&s| s >= SQ)
        || cert.region.windows(2).any(|v| v[0] >= v[1])
        || region(game).as_ref() != Some(&cert.region)
        || cert.nodes[0].fen != cert.root
    {
        return Err("invalid fort certificate root or region".into());
    }
    if game
        .outcome
        .as_ref()
        .is_some_and(|o| o.winner != Some(Side::Defender))
    {
        return Err("terminal loss cannot be a fort".into());
    }
    let mut unique = HashSet::new();
    let mut states = Vec::with_capacity(cert.nodes.len());
    let mut edge_count = 0;
    for node in &cert.nodes {
        check()?;
        edge_count += node.edges.len();
        if edge_count > 2000000 || !unique.insert(&node.fen) {
            return Err("duplicate or oversized fort graph".into());
        }
        let mut state = normalized(Game::from_fen(&node.fen)?);
        state.adjudicate();
        if !safe(&state, &cert.region) {
            return Err("fort invariant violated".into());
        }
        states.push(state);
    }
    let mut reached = HashSet::from([0]);
    let mut queue = VecDeque::from([0]);
    while let Some(i) = queue.pop_front() {
        check()?;
        let state = &states[i];
        let edges = &cert.nodes[i].edges;
        if state.outcome.is_some() {
            if !edges.is_empty() {
                return Err("terminal fort node has moves".into());
            }
            continue;
        }
        let legal: HashSet<String> = state
            .legal_moves()
            .iter()
            .map(ToString::to_string)
            .collect();
        let recorded: HashSet<String> = edges.iter().map(|e| e.mv.clone()).collect();
        if recorded.len() != edges.len()
            || edges.is_empty()
            || (state.side == Side::Attacker && recorded != legal)
            || (state.side == Side::Defender && (edges.len() != 1 || !recorded.is_subset(&legal)))
        {
            return Err("incomplete attacker coverage or invalid defender strategy".into());
        }
        for arc in edges {
            check()?;
            if arc.next >= states.len() {
                return Err("missing fort successor".into());
            }
            let next = transition(state, arc.mv.parse()?);
            if next.fen() != cert.nodes[arc.next].fen || next.outcome != states[arc.next].outcome {
                return Err("forged fort transition".into());
            }
            if reached.insert(arc.next) {
                queue.push_back(arc.next);
            }
        }
    }
    if reached.len() != states.len() {
        return Err("unreachable fort evidence".into());
    }
    Ok(VerifiedFort {
        root: cert.root.clone(),
    })
}

struct Node {
    game: Game,
    edges: Vec<(Move, usize)>,
    expanded: bool,
    safe: bool,
}
fn survivors(nodes: &[Node]) -> Vec<bool> {
    let mut alive: Vec<_> = nodes.iter().map(|n| n.safe && n.expanded).collect();
    let mut parents = vec![Vec::new(); nodes.len()];
    let mut counts: Vec<_> = nodes.iter().map(|n| n.edges.len()).collect();
    for (i, node) in nodes.iter().enumerate() {
        for &(_, next) in &node.edges {
            parents[next].push(i);
        }
    }
    let mut queue: VecDeque<_> = alive
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| (!v).then_some(i))
        .collect();
    while let Some(bad) = queue.pop_front() {
        for &parent in &parents[bad] {
            counts[parent] = counts[parent].saturating_sub(1);
            if alive[parent] && (nodes[parent].game.side == Side::Attacker || counts[parent] == 0) {
                alive[parent] = false;
                queue.push_back(parent);
            }
        }
    }
    alive
}
fn certificate(nodes: &[Node], alive: &[bool], cells: Vec<usize>) -> Certificate {
    let mut selected = vec![0];
    let mut ids = HashMap::from([(0, 0)]);
    let mut output = Vec::new();
    let mut cursor = 0;
    while cursor < selected.len() {
        let node = &nodes[selected[cursor]];
        let mut edges = Vec::new();
        for &(mv, target) in &node.edges {
            if !alive[target] {
                continue;
            }
            let next = *ids.entry(target).or_insert_with(|| {
                selected.push(target);
                selected.len() - 1
            });
            edges.push(ArcMove {
                mv: mv.to_string(),
                next,
            });
            if node.game.side == Side::Defender {
                break;
            }
        }
        output.push(FortNode {
            fen: node.game.fen(),
            edges,
        });
        cursor += 1;
    }
    Certificate {
        version: 1,
        root: nodes[0].game.fen(),
        region: cells,
        nodes: output,
    }
}
pub fn prove(game: &Game, max_positions: usize, time: Duration, stop: &AtomicBool) -> FortProof {
    let start = Instant::now();
    let unknown = |positions, reason: &str| FortProof {
        status: "unknown",
        positions,
        reason: reason.into(),
        certificate: None,
    };
    let Some(cells) = region(game) else {
        return unknown(1, "Unsupported formation: no enclosed edge-king region");
    };
    let mut root = normalized(game.clone());
    root.adjudicate();
    if !safe(&root, &cells) {
        return unknown(1, "King mobility or protected-region invariant fails");
    }
    let mut nodes = vec![Node {
        safe: true,
        game: root,
        edges: vec![],
        expanded: false,
    }];
    let mut ids = HashMap::from([(game.fen(), 0)]);
    let mut cursor = 0;
    let mut total_edges = 0;
    while cursor < nodes.len() {
        if start.elapsed() >= time || stop.load(Ordering::Relaxed) {
            return unknown(nodes.len(), "Time limit or cancellation");
        }
        if nodes[cursor].safe && nodes[cursor].game.outcome.is_none() {
            for mv in nodes[cursor].game.legal_moves() {
                if start.elapsed() >= time || stop.load(Ordering::Relaxed) {
                    return unknown(nodes.len(), "Time limit or cancellation");
                }
                let child = transition(&nodes[cursor].game, mv);
                let key = child.fen();
                let next = if let Some(&i) = ids.get(&key) {
                    i
                } else {
                    if nodes.len() >= max_positions.min(100000) {
                        return unknown(nodes.len(), "Position limit");
                    }
                    let i = nodes.len();
                    ids.insert(key, i);
                    nodes.push(Node {
                        safe: safe(&child, &cells),
                        expanded: child.outcome.is_some() || !safe(&child, &cells),
                        game: child,
                        edges: vec![],
                    });
                    i
                };
                nodes[cursor].edges.push((mv, next));
                total_edges += 1;
                if total_edges > 2000000 {
                    return unknown(nodes.len(), "Transition limit");
                }
            }
        }
        nodes[cursor].expanded = true;
        cursor += 1;
        if cursor.is_multiple_of(32) || cursor == nodes.len() {
            let alive = survivors(&nodes);
            if alive[0] {
                let cert = certificate(&nodes, &alive, cells);
                if start.elapsed() >= time || stop.load(Ordering::Relaxed) {
                    return unknown(nodes.len(), "Time limit or cancellation");
                }
                return match verify_bounded(game, &cert, time.saturating_sub(start.elapsed()), stop)
                {
                    Ok(_) => FortProof {
                        status: "proven",
                        positions: nodes.len(),
                        reason:
                            "Verified closed defensive strategy; every attacker response covered"
                                .into(),
                        certificate: Some(cert),
                    },
                    Err(e) => unknown(
                        nodes.len(),
                        &format!("Certificate verification failed: {e}"),
                    ),
                };
            }
        }
    }
    unknown(
        nodes.len(),
        "No closed strategy satisfying the conservative fort invariant",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greatest_fixed_point_requires_all_attacker_edges_but_one_defender_edge() {
        let attacker = Game {
            side: Side::Attacker,
            ..Game::default()
        };
        let mut defender = attacker.clone();
        defender.side = Side::Defender;
        let mv = Move { from: 3, to: 14 };
        let mut nodes = vec![
            Node {
                game: attacker,
                edges: vec![(mv, 1)],
                expanded: true,
                safe: true,
            },
            Node {
                game: defender,
                edges: vec![(mv, 0), (mv, 2)],
                expanded: true,
                safe: true,
            },
            Node {
                game: Game::default(),
                edges: vec![],
                expanded: true,
                safe: false,
            },
        ];
        assert_eq!(survivors(&nodes), vec![true, true, false]);
        nodes[0].edges.push((mv, 2));
        assert_eq!(survivors(&nodes), vec![false, false, false]);
    }
}
