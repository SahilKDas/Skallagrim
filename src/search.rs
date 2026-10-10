use crate::{
    game::{Game, Move, Piece, Side},
    neural::{Accumulator, Models, policy_index},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bot {
    Rho,
    Iota,
    Beta,
    Baseline,
}
impl std::str::FromStr for Bot {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "rho" => Ok(Self::Rho),
            "iota" => Ok(Self::Iota),
            "beta" => Ok(Self::Beta),
            "baseline" => Ok(Self::Baseline),
            _ => Err("unknown bot".into()),
        }
    }
}
pub struct Limits {
    pub time: Duration,
    pub nodes: u64,
    pub depth: u32,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            time: Duration::from_millis(300),
            nodes: u64::MAX,
            depth: 64,
        }
    }
}
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best: Option<Move>,
    pub value: f32,
    pub nodes: u64,
    pub depth: u32,
    pub elapsed: Duration,
    pub policy: Vec<(Move, f32)>,
}
pub fn terminal(g: &Game) -> Option<f32> {
    g.outcome.as_ref().map(|o| match o.winner {
        Some(Side::Attacker) => 1.0,
        Some(Side::Defender) => -1.0,
        None => 0.0,
    })
}
pub fn handcrafted(g: &Game) -> f32 {
    if let Some(v) = terminal(g) {
        return v;
    }
    let a = g.board.iter().filter(|&&p| p == Piece::Attacker).count() as f32;
    let d = g.board.iter().filter(|&&p| p == Piece::Defender).count() as f32;
    let k = g.king().unwrap();
    let x = k % 11;
    let y = k / 11;
    let distance = x.min(10 - x) + y.min(10 - y);
    ((a - 2.0 * d) * 0.045 + (distance as f32 - 6.0) * 0.08).tanh()
}
pub fn ready(bot: Bot, models: &Models) -> Result<(), String> {
    match bot {
        Bot::Rho if models.cnn.is_none() => {
            Err("Rho requires CNNFile; export/train a CNN first".into())
        }
        Bot::Iota | Bot::Beta if models.nnue.is_none() => {
            Err("Iota/Beta require NNUEFile; export/train an NNUE first".into())
        }
        _ => Ok(()),
    }
}
struct Context<'a> {
    models: &'a Models,
    bot: Bot,
    deadline: Instant,
    stop: Arc<AtomicBool>,
    nodes: u64,
    max_nodes: u64,
}
impl Context<'_> {
    fn expired(&self) -> bool {
        self.nodes >= self.max_nodes
            || Instant::now() >= self.deadline
            || self.stop.load(Ordering::Relaxed)
    }
    fn value(&self, g: &Game, acc: Option<&Accumulator>) -> f32 {
        if let Some(v) = terminal(g) {
            return v;
        }
        if self.bot == Bot::Baseline {
            return handcrafted(g);
        }
        if self.bot == Bot::Rho {
            return self
                .models
                .cnn
                .as_ref()
                .unwrap()
                .infer_cancellable(g, &|| self.expired())
                .map(|result| result.0)
                .filter(|v| v.is_finite())
                .unwrap_or_else(|| handcrafted(g));
        }
        let nn = self.models.nnue.as_ref().unwrap();
        if let Some(a) = acc {
            nn.value(a)
        } else {
            nn.value(&nn.accumulator(g))
        }
    }
    fn alphabeta(
        &mut self,
        g: &Game,
        depth: u32,
        mut alpha: f32,
        beta: f32,
        acc: Option<&Accumulator>,
    ) -> f32 {
        if self.expired() {
            return self.value(g, acc) * g.side.sign();
        }
        self.nodes += 1;
        if depth == 0 || g.outcome.is_some() {
            return self.value(g, acc) * g.side.sign();
        }
        let mut children = Vec::new();
        for m in g.legal_moves() {
            if self.expired() {
                break;
            }
            let mut child = g.clone();
            child.play_unchecked(m);
            let score = terminal(&child).unwrap_or(0.0) * g.side.sign() * 100.0
                + child.captures.len() as f32;
            children.push((child, score));
        }
        children.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut best: f32 = -2.0;
        for (child, _) in children {
            if self.expired() {
                break;
            }
            let next = acc.map(|a| self.models.nnue.as_ref().unwrap().update(a, &child));
            let v = -self.alphabeta(&child, depth - 1, -beta, -alpha, next.as_ref());
            best = best.max(v);
            alpha = alpha.max(v);
            if alpha >= beta {
                break;
            }
        }
        if best == -2.0 {
            self.value(g, acc) * g.side.sign()
        } else {
            best
        }
    }
}
struct Node {
    g: Game,
    acc: Option<Accumulator>,
    visits: u32,
    sum: f32,
    children: Vec<(Move, f32, Option<usize>)>,
    expanded: bool,
}
pub fn search(
    g: &Game,
    bot: Bot,
    models: &Models,
    limits: Limits,
    stop: Arc<AtomicBool>,
) -> Result<SearchResult, String> {
    ready(bot, models)?;
    let start = Instant::now();
    let moves = g.legal_moves();
    if moves.is_empty() {
        return Ok(SearchResult {
            best: None,
            value: terminal(g).unwrap_or(0.0),
            nodes: 0,
            depth: 0,
            elapsed: start.elapsed(),
            policy: vec![],
        });
    }
    let mut ctx = Context {
        models,
        bot,
        deadline: start
            + limits.time.saturating_sub(
                Duration::from_millis(if bot == Bot::Beta { 35 } else { 10 }).min(limits.time / 5),
            ),
        stop,
        nodes: 0,
        max_nodes: limits.nodes,
    };
    let acc = if matches!(bot, Bot::Iota | Bot::Beta) {
        Some(models.nnue.as_ref().unwrap().accumulator(g))
    } else {
        None
    };
    let mut best = moves[0];
    let mut value = ctx.value(g, acc.as_ref());
    let mut completed = 0;
    let mut policy = vec![];
    if matches!(bot, Bot::Iota | Bot::Baseline) {
        let mut ordered = moves.clone();
        for depth in 1..=limits.depth {
            let mut scores = vec![];
            let mut alpha = -2.0;
            for &m in &ordered {
                if ctx.expired() {
                    break;
                }
                let mut child = g.clone();
                child.play_unchecked(m);
                let next = acc
                    .as_ref()
                    .map(|a| models.nnue.as_ref().unwrap().update(a, &child));
                let v = -ctx.alphabeta(&child, depth - 1, -2.0, -alpha, next.as_ref());
                alpha = alpha.max(v);
                scores.push((m, v));
            }
            if ctx.expired() || scores.len() != moves.len() {
                // A short budget may not finish depth one; use evaluated moves.
                if completed == 0 && !scores.is_empty() {
                    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
                    best = scores[0].0;
                    value = scores[0].1 * g.side.sign();
                }
                break;
            }
            scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            best = scores[0].0;
            value = scores[0].1 * g.side.sign();
            completed = depth;
            ordered = scores.into_iter().map(|x| x.0).collect();
        }
        policy = vec![(best, 1.0)];
    } else {
        let mut arena = vec![Node {
            g: g.clone(),
            acc,
            visits: 0,
            sum: 0.0,
            children: vec![],
            expanded: false,
        }];
        'simulations: while !ctx.expired() {
            let mut path = vec![0];
            let mut at = 0;
            while arena[at].expanded
                && !arena[at].children.is_empty()
                && path.len() <= limits.depth as usize
            {
                if ctx.expired() {
                    break 'simulations;
                }
                let n = &arena[at];
                let sqrt = (n.visits.max(1) as f32).sqrt();
                let edge = n
                    .children
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        let score = |e: &(Move, f32, Option<usize>)| {
                            let (q, visits) =
                                e.2.map(|i| {
                                    let c = &arena[i];
                                    (
                                        if c.visits == 0 {
                                            0.0
                                        } else {
                                            c.sum / c.visits as f32 * n.g.side.sign()
                                        },
                                        c.visits,
                                    )
                                })
                                .unwrap_or((0.0, 0));
                            q + 1.5 * e.1 * sqrt / (1 + visits) as f32
                        };
                        score(a).total_cmp(&score(b))
                    })
                    .unwrap()
                    .0;
                let next = if let Some(index) = arena[at].children[edge].2 {
                    index
                } else {
                    let mv = arena[at].children[edge].0;
                    let mut child = arena[at].g.clone();
                    child.play_unchecked(mv);
                    let acc = arena[at]
                        .acc
                        .as_ref()
                        .map(|a| models.nnue.as_ref().unwrap().update(a, &child));
                    let index = arena.len();
                    arena.push(Node {
                        g: child,
                        acc,
                        visits: 0,
                        sum: 0.0,
                        children: vec![],
                        expanded: false,
                    });
                    arena[at].children[edge].2 = Some(index);
                    index
                };
                at = next;
                path.push(at);
            }
            let v;
            if let Some(t) = terminal(&arena[at].g) {
                v = t;
            } else {
                let moves = arena[at].g.legal_moves();
                let logits;
                if bot == Bot::Rho {
                    let Some((val, p)) = models
                        .cnn
                        .as_ref()
                        .unwrap()
                        .infer_cancellable(&arena[at].g, &|| ctx.expired())
                    else {
                        break 'simulations;
                    };
                    v = if val.is_finite() {
                        val
                    } else {
                        handcrafted(&arena[at].g)
                    };
                    logits = moves
                        .iter()
                        .map(|m| {
                            let v = p[policy_index(*m)];
                            if v.is_finite() { v } else { 0.0 }
                        })
                        .collect::<Vec<_>>();
                } else {
                    v = ctx.value(&arena[at].g, arena[at].acc.as_ref());
                    logits = vec![0.0; moves.len()];
                }
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let probabilities: Vec<_> = logits.iter().map(|v| (v - max).exp()).collect();
                let sum: f32 = probabilities.iter().sum();
                // Child states are materialized only when visited. Eager creation
                // copied every history hundreds of times and delayed teardown.
                arena[at].children = if path.len() <= limits.depth as usize {
                    moves
                        .into_iter()
                        .zip(probabilities)
                        .map(|(m, p)| (m, p / sum, None))
                        .collect()
                } else {
                    vec![]
                };
                arena[at].expanded = true;
            }
            ctx.nodes += 1;
            for idx in &path {
                arena[*idx].visits += 1;
                arena[*idx].sum += v;
            }
            completed = completed.max(path.len() as u32);
        }
        if let Some(&(m, _, idx)) = arena[0]
            .children
            .iter()
            .max_by_key(|e| e.2.map(|i| arena[i].visits).unwrap_or(0))
        {
            best = m;
            if let Some(idx) = idx
                && arena[idx].visits > 0
            {
                value = arena[idx].sum / arena[idx].visits as f32;
            }
        }
        let count: u32 = arena[0]
            .children
            .iter()
            .map(|e| e.2.map(|i| arena[i].visits).unwrap_or(0))
            .sum();
        if count > 0 {
            policy = arena[0]
                .children
                .iter()
                .map(|e| {
                    (
                        e.0,
                        e.2.map(|i| arena[i].visits).unwrap_or(0) as f32 / count as f32,
                    )
                })
                .collect();
        } else {
            policy.push((best, 1.0));
        }
    }
    Ok(SearchResult {
        best: Some(best),
        value,
        nodes: ctx.nodes,
        depth: completed,
        elapsed: start.elapsed(),
        policy,
    })
}
