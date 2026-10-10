use crate::game::{Game, Piece, SQ, Side};
use serde::{Deserialize, Serialize};
use std::path::Path;
pub const FEATURES: usize = SQ * 3 + 1;
pub const PLANES: usize = 6;

pub fn features(g: &Game) -> Vec<usize> {
    let mut active = Vec::with_capacity(40);
    for (s, p) in g.board.iter().enumerate() {
        match p {
            Piece::Attacker => active.push(s),
            Piece::Defender => active.push(SQ + s),
            Piece::King => active.push(2 * SQ + s),
            _ => {}
        }
    }
    if g.side == Side::Attacker {
        active.push(3 * SQ);
    }
    active
}
pub fn planes(g: &Game) -> Vec<f32> {
    let mut x = vec![0.0; PLANES * SQ];
    for f in features(g) {
        if f < 3 * SQ {
            x[f] = 1.0;
        }
    }
    let fen = g.fen();
    let repeats = g
        .history
        .iter()
        .filter(|f| f.as_ref() == fen)
        .count()
        .min(2) as f32
        / 2.0;
    for s in 0..SQ {
        x[3 * SQ + s] = if g.side == Side::Attacker { 1.0 } else { 0.0 };
        x[4 * SQ + s] = if crate::game::corner(s) || s == 60 {
            1.0
        } else {
            0.0
        };
        x[5 * SQ + s] = repeats;
    }
    x
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Nnue {
    pub format: String,
    pub hidden: usize,
    /// Input-major weights permit contiguous accumulator updates.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub output: Vec<f32>,
    pub output_bias: f32,
}
#[derive(Clone)]
pub struct Accumulator {
    pub values: Vec<f32>,
    pub active: Vec<usize>,
}
impl Nnue {
    pub fn validate(&self) -> Result<(), String> {
        if self.format != "skallagrim-nnue-v1"
            || self.hidden == 0
            || self.hidden > 1024
            || self.weights.len() != FEATURES * self.hidden
            || self.bias.len() != self.hidden
            || self.output.len() != self.hidden
        {
            return Err("invalid NNUE shape/format".into());
        }
        if self
            .weights
            .iter()
            .chain(&self.bias)
            .chain(&self.output)
            .any(|x| !x.is_finite())
            || !self.output_bias.is_finite()
        {
            return Err("non-finite NNUE weights".into());
        }
        Ok(())
    }
    pub fn accumulator(&self, g: &Game) -> Accumulator {
        let active = features(g);
        let mut values = self.bias.clone();
        for &f in &active {
            for (j, v) in values.iter_mut().enumerate() {
                *v += self.weights[f * self.hidden + j];
            }
        }
        Accumulator { values, active }
    }
    pub fn update(&self, previous: &Accumulator, g: &Game) -> Accumulator {
        let active = features(g);
        let mut values = previous.values.clone();
        for &f in &previous.active {
            if !active.contains(&f) {
                for (j, v) in values.iter_mut().enumerate() {
                    *v -= self.weights[f * self.hidden + j];
                }
            }
        }
        for &f in &active {
            if !previous.active.contains(&f) {
                for (j, v) in values.iter_mut().enumerate() {
                    *v += self.weights[f * self.hidden + j];
                }
            }
        }
        Accumulator { values, active }
    }
    pub fn value(&self, a: &Accumulator) -> f32 {
        (self.output_bias
            + a.values
                .iter()
                .zip(&self.output)
                .map(|(x, w)| x.clamp(0.0, 1.0) * w)
                .sum::<f32>())
        .tanh()
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Conv {
    pub input: usize,
    pub output: usize,
    pub kernel: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}
impl Conv {
    fn validate(&self) -> bool {
        (self.kernel == 1 || self.kernel == 3)
            && self.input > 0
            && self.output > 0
            && self.weights.len() == self.input * self.output * self.kernel * self.kernel
            && self.bias.len() == self.output
            && self.weights.iter().chain(&self.bias).all(|v| v.is_finite())
    }
    fn run(&self, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.output * SQ];
        let pad = (self.kernel / 2) as i32;
        for oc in 0..self.output {
            for y in 0..11 {
                for x in 0..11 {
                    let mut v = self.bias[oc];
                    for ic in 0..self.input {
                        for ky in 0..self.kernel {
                            for kx in 0..self.kernel {
                                let (xx, yy) =
                                    (x as i32 + kx as i32 - pad, y as i32 + ky as i32 - pad);
                                if (0..11).contains(&xx) && (0..11).contains(&yy) {
                                    v += input[ic * SQ + yy as usize * 11 + xx as usize]
                                        * self.weights[((oc * self.input + ic) * self.kernel + ky)
                                            * self.kernel
                                            + kx];
                                }
                            }
                        }
                    }
                    out[oc * SQ + y * 11 + x] = v;
                }
            }
        }
        out
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Cnn {
    pub format: String,
    pub channels: usize,
    pub stem: Conv,
    pub blocks: Vec<[Conv; 2]>,
    pub policy: Conv,
    pub value: Vec<f32>,
    pub value_bias: f32,
}
impl Cnn {
    pub fn validate(&self) -> Result<(), String> {
        let c = self.channels;
        if self.format != "skallagrim-cnn-v1"
            || c == 0
            || c > 128
            || self.blocks.len() > 32
            || !self.stem.validate()
            || self.stem.input != PLANES
            || self.stem.output != c
            || self.stem.kernel != 3
            || !self.policy.validate()
            || self.policy.input != c
            || self.policy.output != 40
            || self.policy.kernel != 1
            || self.value.len() != c
            || self.value.iter().any(|v| !v.is_finite())
            || !self.value_bias.is_finite()
            || self
                .blocks
                .iter()
                .flatten()
                .any(|b| !b.validate() || b.input != c || b.output != c || b.kernel != 3)
        {
            return Err("invalid CNN shape/format".into());
        }
        Ok(())
    }
    pub fn infer(&self, g: &Game) -> (f32, Vec<f32>) {
        let mut x = self.stem.run(&planes(g));
        for v in &mut x {
            *v = v.max(0.0);
        }
        for block in &self.blocks {
            let mut y = block[0].run(&x);
            for v in &mut y {
                *v = v.max(0.0);
            }
            let y = block[1].run(&y);
            for (a, b) in x.iter_mut().zip(y) {
                *a = (*a + b).max(0.0);
            }
        }
        let v = self.value_bias
            + (0..self.channels)
                .map(|c| x[c * SQ..(c + 1) * SQ].iter().sum::<f32>() / SQ as f32 * self.value[c])
                .sum::<f32>();
        (v.tanh(), self.policy.run(&x))
    }
}
pub fn policy_index(m: crate::game::Move) -> usize {
    let (dx, dy) = (
        (m.to % 11) as i32 - (m.from % 11) as i32,
        (m.to / 11) as i32 - (m.from / 11) as i32,
    );
    let (dir, distance) = if dx > 0 {
        (0, dx)
    } else if dx < 0 {
        (1, -dx)
    } else if dy > 0 {
        (2, dy)
    } else {
        (3, -dy)
    };
    (dir * 10 + distance as usize - 1) * SQ + m.from
}
#[derive(Clone, Default)]
pub struct Models {
    pub nnue: Option<Nnue>,
    pub cnn: Option<Cnn>,
}
impl Models {
    pub fn load_nnue(&mut self, path: &Path) -> Result<(), String> {
        let n: Nnue = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        n.validate()?;
        self.nnue = Some(n);
        Ok(())
    }
    pub fn load_cnn(&mut self, path: &Path) -> Result<(), String> {
        let n: Cnn = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        n.validate()?;
        self.cnn = Some(n);
        Ok(())
    }
}
