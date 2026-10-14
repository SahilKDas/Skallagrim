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
    pub deadline_fallback: bool,
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
/// Includes command handling and worker startup in the response budget.
pub fn response_deadline(start: Instant, budget: Duration) -> Instant {
    start + budget.saturating_sub(Duration::from_millis(10).min(budget / 5))
}

fn supervise(
    mut fallback: SearchResult,
    deadline: Instant,
    stop: Arc<AtomicBool>,
    compute: impl FnOnce(std::sync::mpsc::Sender<Result<SearchResult, String>>) + Send + 'static,
    publish: impl FnOnce(Result<SearchResult, String>),
) {
    let start = Instant::now();
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || compute(tx));
    let response_priority = ResponsePriority::new();
    let result = match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        Ok(result) => result,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            stop.store(true, Ordering::Relaxed);
            fallback.elapsed = start.elapsed();
            Ok(fallback)
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Err("search worker ended without a result".into())
        }
    };
    // Exactly one publication, then reap the worker. Cleanup never precedes reply.
    publish(result);
    drop(response_priority);
    let _ = worker.join();
}

// The sleeping response thread must preempt compute/cleanup on a pinned Windows
// core. Only the brief publication runs above normal; search keeps normal priority.
struct ResponsePriority {
    #[cfg(windows)]
    previous: Option<i32>,
}
#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentThread() -> *mut std::ffi::c_void;
    fn GetThreadPriority(thread: *mut std::ffi::c_void) -> i32;
    fn SetThreadPriority(thread: *mut std::ffi::c_void, priority: i32) -> i32;
}
impl ResponsePriority {
    fn new() -> Self {
        Self {
            #[cfg(windows)]
            previous: unsafe {
                let thread = GetCurrentThread();
                let old = GetThreadPriority(thread);
                if old != i32::MAX && SetThreadPriority(thread, old.max(1)) != 0 {
                    Some(old)
                } else {
                    None
                }
            },
        }
    }
}
impl Drop for ResponsePriority {
    fn drop(&mut self) {
        #[cfg(windows)]
        if let Some(old) = self.previous {
            unsafe {
                SetThreadPriority(GetCurrentThread(), old);
            }
        }
    }
}

pub fn search_with_deadline(
    g: Game,
    bot: Bot,
    models: Models,
    limits: Limits,
    stop: Arc<AtomicBool>,
    deadline: Instant,
    publish: impl FnOnce(Result<SearchResult, String>),
) {
    if let Err(error) = ready(bot, &models) {
        publish(Err(error));
        return;
    }
    let best = g.legal_moves().first().copied();
    let fallback = SearchResult {
        best,
        value: terminal(&g).unwrap_or_else(|| handcrafted(&g)),
        nodes: 0,
        depth: 0,
        elapsed: Duration::ZERO,
        policy: best.map(|mv| vec![(mv, 1.0)]).unwrap_or_default(),
        deadline_fallback: true,
    };
    let flag = stop.clone();
    supervise(
        fallback,
        deadline,
        stop,
        move |tx| {
            let finish_by = deadline
                .checked_sub(Duration::from_millis(5).min(limits.time / 10))
                .unwrap_or(deadline);
            let result = search_publishing(
                &g,
                bot,
                &models,
                limits,
                flag,
                Some(finish_by),
                &mut |result| {
                    let _ = tx.send(Ok(result.clone()));
                    std::thread::yield_now();
                },
            );
            if let Err(error) = result {
                let _ = tx.send(Err(error));
            }
        },
        publish,
    );
}
pub fn search(
    g: &Game,
    bot: Bot,
    models: &Models,
    limits: Limits,
    stop: Arc<AtomicBool>,
) -> Result<SearchResult, String> {
    search_publishing(g, bot, models, limits, stop, None, &mut |_| {})
}
fn search_publishing(
    g: &Game,
    bot: Bot,
    models: &Models,
    limits: Limits,
    stop: Arc<AtomicBool>,
    finish_by: Option<Instant>,
    publish: &mut dyn FnMut(&SearchResult),
) -> Result<SearchResult, String> {
    ready(bot, models)?;
    let start = Instant::now();
    let moves = g.legal_moves();
    if moves.is_empty() {
        let result = SearchResult {
            best: None,
            value: terminal(g).unwrap_or(0.0),
            nodes: 0,
            depth: 0,
            elapsed: start.elapsed(),
            policy: vec![],
            deadline_fallback: false,
        };
        publish(&result);
        return Ok(result);
    }
    let mut ctx = Context {
        models,
        bot,
        deadline: (start
            + limits.time.saturating_sub(
                Duration::from_millis(if bot == Bot::Beta { 35 } else { 10 }).min(limits.time / 5),
            ))
        .min(finish_by.unwrap_or(start + limits.time)),
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
    let mut arena = Vec::new();
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
        arena.push(Node {
            g: g.clone(),
            acc,
            visits: 0,
            sum: 0.0,
            children: vec![],
            expanded: false,
        });
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
    let result = SearchResult {
        best: Some(best),
        value,
        nodes: ctx.nodes,
        depth: completed,
        elapsed: start.elapsed(),
        policy,
        deadline_fallback: false,
    };
    // Publish while the arena is still alive: deallocation is not move latency.
    publish(&result);
    Ok(result)
}

#[cfg(test)]
mod deadline_tests {
    use super::*;

    fn fallback() -> SearchResult {
        let mv = Game::default().legal_moves()[0];
        SearchResult {
            best: Some(mv),
            value: 0.0,
            nodes: 0,
            depth: 0,
            elapsed: Duration::ZERO,
            policy: vec![(mv, 1.0)],
            deadline_fallback: true,
        }
    }

    #[test]
    fn stalled_worker_publishes_once_cancels_and_reaps_late_worker() {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let completed = Arc::new(AtomicBool::new(false));
        let done = completed.clone();
        let mut publications = 0;
        supervise(
            fallback(),
            Instant::now() + Duration::from_millis(20),
            stop.clone(),
            move |tx| {
                while !flag.load(Ordering::Relaxed) {
                    std::thread::yield_now();
                }
                let _ = tx.send(Ok(fallback()));
                done.store(true, Ordering::Relaxed);
            },
            |result| {
                publications += 1;
                let result = result.unwrap();
                assert!(
                    Game::default()
                        .legal_moves()
                        .contains(&result.best.unwrap())
                );
                assert_eq!(result.policy, vec![(result.best.unwrap(), 1.0)]);
            },
        );
        assert_eq!(publications, 1);
        assert!(stop.load(Ordering::Relaxed));
        assert!(completed.load(Ordering::Relaxed));
    }

    #[test]
    fn completed_result_is_published_before_cleanup_and_keeps_policy() {
        let published = Arc::new(AtomicBool::new(false));
        let signal = published.clone();
        let cleaned = Arc::new(AtomicBool::new(false));
        let done = cleaned.clone();
        supervise(
            fallback(),
            Instant::now() + Duration::from_secs(2),
            Arc::new(AtomicBool::new(false)),
            move |tx| {
                let mut result = fallback();
                result.nodes = 42;
                tx.send(Ok(result)).unwrap();
                let watchdog = Instant::now() + Duration::from_secs(3);
                while !signal.load(Ordering::Relaxed) {
                    assert!(Instant::now() < watchdog);
                    std::thread::yield_now();
                }
                done.store(true, Ordering::Relaxed);
            },
            |result| {
                assert_eq!(result.unwrap().nodes, 42);
                assert!(!cleaned.load(Ordering::Relaxed));
                published.store(true, Ordering::Relaxed);
            },
        );
        assert!(cleaned.load(Ordering::Relaxed));
    }

    #[test]
    fn terminal_deadline_result_has_no_move() {
        let g = Game {
            outcome: Some(crate::game::Outcome {
                winner: Some(Side::Defender),
                reason: "test".into(),
            }),
            ..Game::default()
        };
        search_with_deadline(
            g,
            Bot::Baseline,
            Models::default(),
            Limits::default(),
            Arc::new(AtomicBool::new(false)),
            Instant::now(),
            |result| {
                let result = result.unwrap();
                assert!(result.best.is_none());
                assert!(result.policy.is_empty());
            },
        );
    }

    #[cfg(windows)]
    #[test]
    fn response_priority_is_scoped_and_compute_stays_normal() {
        let before = unsafe { GetThreadPriority(GetCurrentThread()) };
        supervise(
            fallback(),
            Instant::now() + Duration::from_secs(2),
            Arc::new(AtomicBool::new(false)),
            |tx| {
                assert_eq!(unsafe { GetThreadPriority(GetCurrentThread()) }, 0);
                tx.send(Ok(fallback())).unwrap();
            },
            |result| {
                assert!(result.is_ok());
                assert_eq!(
                    unsafe { GetThreadPriority(GetCurrentThread()) },
                    before.max(1)
                );
            },
        );
        assert_eq!(unsafe { GetThreadPriority(GetCurrentThread()) }, before);
    }
}
