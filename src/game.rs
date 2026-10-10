//! Copenhagen rules, independently implemented from the federation's prose.
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, sync::Arc};
pub const RULES_VERSION: &str = "copenhagen-static-v3";
pub const N: usize = 11;
pub const SQ: usize = 121;
pub const THRONE: usize = 60;
pub const START: &str = "3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Attacker,
    Defender,
}
impl Side {
    pub fn other(self) -> Self {
        if self == Self::Attacker {
            Self::Defender
        } else {
            Self::Attacker
        }
    }
    pub fn sign(self) -> f32 {
        if self == Self::Attacker { 1.0 } else { -1.0 }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Piece {
    Empty,
    Attacker,
    Defender,
    King,
}
impl Piece {
    pub fn side(self) -> Option<Side> {
        match self {
            Self::Empty => None,
            Self::Attacker => Some(Side::Attacker),
            _ => Some(Side::Defender),
        }
    }
    pub fn code(self) -> char {
        match self {
            Self::Empty => '.',
            Self::Attacker => 'a',
            Self::Defender => 'd',
            Self::King => 'k',
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Move {
    pub from: usize,
    pub to: usize,
}
pub fn square(s: &str) -> Result<usize, String> {
    let b = s.as_bytes();
    if !(2..=3).contains(&b.len())
        || !(b'a'..=b'k').contains(&b[0])
        || b[1] == b'0'
        || !b[1..].iter().all(u8::is_ascii_digit)
    {
        return Err("invalid square".into());
    }
    let row = s[1..].parse::<usize>().map_err(|_| "invalid rank")?;
    if !(1..=11).contains(&row) {
        return Err("rank outside 1..11".into());
    }
    Ok((row - 1) * 11 + (b[0] - b'a') as usize)
}
pub fn coord(s: usize) -> String {
    format!("{}{}", (b'a' + (s % 11) as u8) as char, s / 11 + 1)
}
impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", coord(self.from), coord(self.to))
    }
}
impl std::str::FromStr for Move {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if !s.is_ascii() {
            return Err("move must be ASCII".into());
        }
        let split = s
            .as_bytes()
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, c)| c.is_ascii_alphabetic())
            .map(|(i, _)| i)
            .ok_or("expected from/to coordinates")?;
        Ok(Self {
            from: square(&s[..split])?,
            to: square(&s[split..])?,
        })
    }
}
pub fn corner(s: usize) -> bool {
    matches!(s, 0 | 10 | 110 | 120)
}
pub fn edge(s: usize) -> bool {
    s.is_multiple_of(11) || s % 11 == 10 || s / 11 == 0 || s / 11 == 10
}
pub fn step(s: usize, dx: i32, dy: i32) -> Option<usize> {
    let (x, y) = ((s % 11) as i32 + dx, (s / 11) as i32 + dy);
    if (0..11).contains(&x) && (0..11).contains(&y) {
        Some((y * 11 + x) as usize)
    } else {
        None
    }
}
pub const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub winner: Option<Side>,
    pub reason: String,
}
#[derive(Clone)]
pub struct Game {
    pub board: [Piece; SQ],
    pub side: Side,
    pub outcome: Option<Outcome>,
    pub history: History,
    pub last_move: Option<Move>,
    pub captures: Vec<usize>,
}
/// Persistent history: cloning a search position never copies its entire past.
/// Iteration is newest first; serialization retains chronological protocol order.
#[derive(Clone, Default)]
pub struct History {
    head: Option<Arc<HistoryEntry>>,
    len: usize,
}
struct HistoryEntry {
    fen: Arc<str>,
    previous: Option<Arc<HistoryEntry>>,
}
impl Drop for HistoryEntry {
    fn drop(&mut self) {
        let mut previous = self.previous.take();
        while let Some(entry) = previous {
            match Arc::try_unwrap(entry) {
                Ok(mut unique) => previous = unique.previous.take(),
                Err(_) => break,
            }
        }
    }
}
impl History {
    pub fn push(&mut self, fen: Arc<str>) {
        self.head = Some(Arc::new(HistoryEntry {
            fen,
            previous: self.head.take(),
        }));
        self.len += 1;
    }
    pub fn iter(&self) -> impl Iterator<Item = &Arc<str>> {
        std::iter::successors(self.head.as_deref(), |entry| entry.previous.as_deref())
            .map(|entry| &entry.fen)
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
impl From<Vec<Arc<str>>> for History {
    fn from(entries: Vec<Arc<str>>) -> Self {
        let mut history = Self::default();
        for entry in entries {
            history.push(entry);
        }
        history
    }
}
impl Serialize for History {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut entries: Vec<_> = self.iter().collect();
        entries.reverse();
        entries.serialize(serializer)
    }
}
impl Default for Game {
    fn default() -> Self {
        Self::from_fen(START).unwrap()
    }
}
impl Game {
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parts: Vec<_> = fen.split_whitespace().collect();
        if parts.len() != 2 {
            return Err("FEN needs board and a/d side".into());
        }
        let side = match parts[1] {
            "a" => Side::Attacker,
            "d" => Side::Defender,
            _ => return Err("invalid side".into()),
        };
        let rows: Vec<_> = parts[0].split('/').collect();
        if rows.len() != 11 {
            return Err("FEN needs eleven ranks (rank 1 first)".into());
        }
        let mut board = [Piece::Empty; SQ];
        for (y, row) in rows.iter().enumerate() {
            if !row
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, 'a' | 'd' | 'k'))
            {
                return Err("invalid piece".into());
            }
            let mut x = 0;
            let mut digits = String::new();
            for c in row.chars().chain(std::iter::once('!')) {
                if c.is_ascii_digit() {
                    digits.push(c);
                    continue;
                }
                if !digits.is_empty() {
                    let n = digits.parse::<usize>().map_err(|_| "invalid empty count")?;
                    if n == 0 || n > 11 {
                        return Err("invalid empty count".into());
                    }
                    x += n;
                    digits.clear();
                }
                if c == '!' {
                    break;
                }
                if x >= 11 {
                    return Err("rank too long".into());
                }
                board[y * 11 + x] = match c {
                    'a' => Piece::Attacker,
                    'd' => Piece::Defender,
                    'k' => Piece::King,
                    _ => return Err("invalid piece".into()),
                };
                x += 1;
            }
            if x != 11 {
                return Err("rank must contain eleven squares".into());
            }
        }
        if board.iter().filter(|&&p| p == Piece::King).count() != 1 {
            return Err("position requires exactly one king".into());
        }
        for (s, p) in board.iter().enumerate() {
            if (corner(s) || s == THRONE) && matches!(p, Piece::Attacker | Piece::Defender) {
                return Err("non-king on restricted square".into());
            }
        }
        let mut game = Self {
            board,
            side,
            outcome: None,
            history: History::default(),
            last_move: None,
            captures: vec![],
        };
        game.history.push(game.fen().into());
        game.adjudicate();
        Ok(game)
    }
    pub fn fen(&self) -> String {
        let mut ranks = vec![];
        for row in self.board.chunks(11) {
            let mut s = String::new();
            let mut n = 0;
            for p in row {
                if *p == Piece::Empty {
                    n += 1;
                } else {
                    if n > 0 {
                        s.push_str(&n.to_string());
                        n = 0;
                    }
                    s.push(p.code());
                }
            }
            if n > 0 {
                s.push_str(&n.to_string());
            }
            ranks.push(s);
        }
        format!(
            "{} {}",
            ranks.join("/"),
            if self.side == Side::Attacker {
                "a"
            } else {
                "d"
            }
        )
    }
    pub fn king(&self) -> Option<usize> {
        self.board.iter().position(|&p| p == Piece::King)
    }
    pub fn legal_moves(&self) -> Vec<Move> {
        if self.outcome.is_some() {
            vec![]
        } else {
            self.moves_for(self.side)
        }
    }
    pub fn moves_for(&self, side: Side) -> Vec<Move> {
        let mut moves = Vec::with_capacity(200);
        for from in 0..SQ {
            let p = self.board[from];
            if p.side() != Some(side) {
                continue;
            }
            for (dx, dy) in DIRS {
                let mut s = from;
                while let Some(to) = step(s, dx, dy) {
                    if self.board[to] != Piece::Empty {
                        break;
                    }
                    if p == Piece::King || (!corner(to) && to != THRONE) {
                        moves.push(Move { from, to });
                    }
                    s = to;
                }
            }
        }
        moves
    }
    fn hostile(&self, s: usize, victim: Side) -> bool {
        corner(s) || (s == THRONE && (victim == Side::Attacker || self.board[s] == Piece::Empty))
    }
    pub fn play(&mut self, mv: Move) -> Result<(), String> {
        if !self.legal_moves().contains(&mv) {
            return Err(format!("illegal move {mv}"));
        }
        self.play_unchecked(mv);
        Ok(())
    }
    pub fn play_unchecked(&mut self, mv: Move) {
        let mover = self.side;
        let enemy = mover.other();
        self.board[mv.to] = self.board[mv.from];
        self.board[mv.from] = Piece::Empty;
        self.captures.clear();
        self.last_move = Some(mv);
        for (dx, dy) in DIRS {
            if let Some(v) = step(mv.to, dx, dy)
                && self.board[v].side() == Some(enemy)
                && self.board[v] != Piece::King
                && let Some(a) = step(v, dx, dy)
                && (self.board[a].side() == Some(mover) || self.hostile(a, enemy))
            {
                self.captures.push(v);
            }
        }
        // Shieldwall membership is evaluated before removing ordinary captures.
        for e in 0..4 {
            let edge_sq = |i: usize| match e {
                0 => i,
                1 => 110 + i,
                2 => i * 11,
                _ => i * 11 + 10,
            };
            let inward = |s| match e {
                0 => s + 11,
                1 => s - 11,
                2 => s + 1,
                _ => s - 1,
            };
            let bracket = |s: usize| self.board[s].side() == Some(mover) || corner(s);
            let mut i = 1;
            while i < 10 {
                if self.board[edge_sq(i)].side() != Some(enemy) {
                    i += 1;
                    continue;
                }
                let first = i;
                while i < 11 && self.board[edge_sq(i)].side() == Some(enemy) {
                    i += 1;
                }
                let end = i;
                if end - first >= 2
                    && end < 11
                    && bracket(edge_sq(first - 1))
                    && bracket(edge_sq(end))
                    && (first..end).all(|j| self.board[inward(edge_sq(j))].side() == Some(mover))
                    && (mv.to == edge_sq(first - 1) || mv.to == edge_sq(end))
                {
                    for j in first..end {
                        let s = edge_sq(j);
                        if self.board[s] != Piece::King {
                            self.captures.push(s);
                        }
                    }
                }
            }
        }
        self.captures.sort_unstable();
        self.captures.dedup();
        for &s in &self.captures {
            self.board[s] = Piece::Empty;
        }
        if mover == Side::Attacker
            && let Some(k) = self.king()
            && !edge(k)
            && DIRS.iter().any(|&(dx, dy)| step(k, dx, dy) == Some(mv.to))
            && DIRS.iter().all(|&(dx, dy)| {
                let s = step(k, dx, dy).unwrap();
                self.board[s] == Piece::Attacker || s == THRONE
            })
        {
            self.board[k] = Piece::Empty;
            self.captures.push(k);
            self.finish(Some(Side::Attacker), "king-captured");
        }
        self.side = enemy;
        // Captured material never returns, so pre-capture positions cannot repeat.
        if !self.captures.is_empty() {
            self.history = History::default();
        }
        self.history.push(self.fen().into());
        self.adjudicate();
    }
    fn finish(&mut self, winner: Option<Side>, reason: &str) {
        self.outcome = Some(Outcome {
            winner,
            reason: reason.into(),
        });
    }
    /// Record a referee decision for the federation's qualitative rules.
    /// Automatic detectors are sufficient proofs, not a general game solver.
    pub fn referee_decision(&mut self, decision: &str) -> Result<(), String> {
        if self.outcome.is_some() {
            return Err("game already finished".into());
        }
        match decision {
            "draw" => self.finish(None, "impossible-to-finish-referee"),
            "fort" => {
                let king = self.king().ok_or("king missing")?;
                if !edge(king)
                    || !self
                        .moves_for(Side::Defender)
                        .iter()
                        .any(|m| m.from == king)
                {
                    return Err("exit fort requires an edge king able to move".into());
                }
                self.finish(Some(Side::Defender), "exit-fort-referee");
            }
            _ => return Err("adjudicate draw|fort".into()),
        }
        Ok(())
    }
    pub fn adjudicate(&mut self) {
        if self.outcome.is_some() {
            return;
        }
        let Some(k) = self.king() else {
            self.finish(Some(Side::Attacker), "king-captured");
            return;
        };
        if corner(k) {
            self.finish(Some(Side::Defender), "corner-escape");
            return;
        }
        if self.encircled() {
            self.finish(Some(Side::Attacker), "encirclement");
            return;
        }
        if self.exit_fort() {
            self.finish(Some(Side::Defender), "exit-fort");
            return;
        }
        let fen = self.fen();
        // Operational definition: third occurrence of identical board + side to move.
        if self.history.iter().filter(|s| s.as_ref() == fen).count() >= 3 {
            self.finish(Some(Side::Attacker), "perpetual-repetition");
            return;
        }
        if self.moves_for(self.side).is_empty() {
            self.finish(Some(self.side.other()), "immobility");
        }
    }
    pub fn encircled(&self) -> bool {
        let mut reached = [false; SQ];
        let mut queue = VecDeque::new();
        for (s, seen) in reached.iter_mut().enumerate() {
            if edge(s) && self.board[s] != Piece::Attacker {
                *seen = true;
                queue.push_back(s);
            }
        }
        while let Some(s) = queue.pop_front() {
            for (dx, dy) in DIRS {
                if let Some(t) = step(s, dx, dy)
                    && !reached[t]
                    && self.board[t] != Piece::Attacker
                {
                    reached[t] = true;
                    queue.push_back(t);
                }
            }
        }
        self.board
            .iter()
            .enumerate()
            .filter(|(_, p)| p.side() == Some(Side::Defender))
            .all(|(s, _)| !reached[s])
    }
    /// Sufficient static proof: an attacker-free king region behind walls that
    /// survive an overapproximation of attacker movement and captures.
    /// Conservative: some valid dynamic forts are not recognized (see RULES.md).
    pub fn exit_fort(&self) -> bool {
        let Some(k) = self.king() else {
            return false;
        };
        if !edge(k) || !self.moves_for(Side::Defender).iter().any(|m| m.from == k) {
            return false;
        }
        let attackers = self.board.iter().filter(|&&p| p == Piece::Attacker).count();
        let mut wall = self.board.map(|p| p == Piece::Defender);
        loop {
            let mut inside = [false; SQ];
            let mut q = VecDeque::from([k]);
            inside[k] = true;
            while let Some(s) = q.pop_front() {
                if self.board[s] == Piece::Attacker {
                    return false;
                }
                for (dx, dy) in DIRS {
                    if let Some(t) = step(s, dx, dy)
                        && !wall[t]
                        && !inside[t]
                    {
                        inside[t] = true;
                        q.push_back(t);
                    }
                }
            }
            // Attackers cannot teleport into disconnected empty pockets. Flood
            // their reachable space while keeping the proposed defender walls
            // fixed. Ignoring other attackers as blockers is an overapproximation.
            let mut reachable = [false; SQ];
            let mut attack_queue = VecDeque::new();
            for (s, p) in self.board.iter().enumerate() {
                if *p == Piece::Attacker {
                    reachable[s] = true;
                    attack_queue.push_back(s);
                }
            }
            while let Some(s) = attack_queue.pop_front() {
                for (dx, dy) in DIRS {
                    if let Some(t) = step(s, dx, dy)
                        && !wall[t]
                        && !inside[t]
                        && !corner(t)
                        && !reachable[t]
                    {
                        reachable[t] = true;
                        attack_queue.push_back(t);
                    }
                }
            }
            // The throne may be crossed but cannot support an attacker.
            let outside = |s: usize| reachable[s] && s != THRONE;
            let mut remove = vec![];
            for (s, &is_wall) in wall.iter().enumerate() {
                if !is_wall {
                    continue;
                }
                for (dx, dy) in [(1, 0), (0, 1)] {
                    if let (Some(a), Some(b)) = (step(s, dx, dy), step(s, -dx, -dy)) {
                        let hostile_a = corner(a) || a == THRONE && self.board[a] == Piece::Empty;
                        let hostile_b = corner(b) || b == THRONE && self.board[b] == Piece::Empty;
                        if (outside(a) || hostile_a)
                            && (outside(b) || hostile_b)
                            && usize::from(!hostile_a) + usize::from(!hostile_b) <= attackers
                        {
                            remove.push(s);
                        }
                    }
                }
            }
            // Overapproximate shieldwalls against contiguous defender edge runs.
            for e in 0..4 {
                let es = |i| match e {
                    0 => i,
                    1 => 110 + i,
                    2 => i * 11,
                    _ => i * 11 + 10,
                };
                let inward = |s| match e {
                    0 => s + 11,
                    1 => s - 11,
                    2 => s + 1,
                    _ => s - 1,
                };
                let mut i = 1;
                while i < 10 {
                    if !wall[es(i)] {
                        i += 1;
                        continue;
                    }
                    let start = i;
                    while i < 11 && wall[es(i)] {
                        i += 1;
                    }
                    if i - start >= 2
                        && i < 11
                        && (outside(es(start - 1)) || corner(es(start - 1)))
                        && (outside(es(i)) || corner(es(i)))
                        && (start..i).all(|j| outside(inward(es(j))))
                        && i - start
                            + usize::from(!corner(es(start - 1)))
                            + usize::from(!corner(es(i)))
                            <= attackers
                    {
                        remove.extend((start..i).map(es));
                    }
                }
            }
            if remove.is_empty() {
                return true;
            }
            for s in remove {
                wall[s] = false;
            }
        }
    }
    pub fn perft(&self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        self.legal_moves()
            .into_iter()
            .map(|m| {
                let mut g = self.clone();
                g.play_unchecked(m);
                g.perft(depth - 1)
            })
            .sum()
    }
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({"fen": self.fen(), "side": self.side, "outcome": self.outcome, "moves": self.legal_moves().iter().map(ToString::to_string).collect::<Vec<_>>(), "history": self.history, "captures": self.captures, "rules": RULES_VERSION})
    }
}
