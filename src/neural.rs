use crate::game::{Game, Piece, SQ, Side};
use serde::{Deserialize, Serialize};
use std::path::Path;
pub const FEATURES: usize = SQ * 3 + 1;
pub const EXPERIMENTAL_FEATURES: usize = FEATURES + 45;
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

pub fn experimental_features(g: &Game) -> Vec<usize> {
    let mut active = features(g);
    let Some(king) = g.board.iter().position(|p| *p == Piece::King) else {
        return active;
    };
    let (y, x) = (king / 11, king % 11);
    let adjacent = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        .iter()
        .filter(|&&(dy, dx)| {
            let (yy, xx) = (y as i32 + dy, x as i32 + dx);
            (0..11).contains(&yy)
                && (0..11).contains(&xx)
                && g.board[yy as usize * 11 + xx as usize] == Piece::Attacker
        })
        .count();
    let mut rays = 0;
    for (bit, (dy, dx)) in [(-1, 0), (1, 0), (0, -1), (0, 1)].into_iter().enumerate() {
        let (mut yy, mut xx) = (y as i32 + dy, x as i32 + dx);
        let mut clear = true;
        while (0..11).contains(&yy) && (0..11).contains(&xx) {
            if g.board[yy as usize * 11 + xx as usize] != Piece::Empty {
                clear = false;
            }
            yy += dy;
            xx += dx;
        }
        let end = ((yy - dy) as usize, (xx - dx) as usize);
        if clear && [(0, 0), (0, 10), (10, 0), (10, 10)].contains(&end) && end != (y, x) {
            rays |= 1 << bit;
        }
    }
    let fen = g.fen();
    let repeats = g
        .history
        .iter()
        .filter(|f| f.as_ref() == fen)
        .count()
        .min(2);
    let distance = [y + x, y + 10 - x, 10 - y + x, 20 - y - x]
        .into_iter()
        .min()
        .unwrap();
    active.extend([
        FEATURES + repeats,
        FEATURES + 3 + distance,
        FEATURES + 24 + adjacent,
        FEATURES + 29 + rays,
    ]);
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
    pub fn active(&self, g: &Game) -> Vec<usize> {
        if self.format == "skallagrim-nnue-v2" {
            experimental_features(g)
        } else {
            features(g)
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        let count = match self.format.as_str() {
            "skallagrim-nnue-v1" => FEATURES,
            "skallagrim-nnue-v2" => EXPERIMENTAL_FEATURES,
            _ => return Err("invalid NNUE format".into()),
        };
        if self.hidden == 0
            || self.hidden > 1024
            || self.weights.len() != count * self.hidden
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
        let active = self.active(g);
        let mut values = self.bias.clone();
        for &f in &active {
            for (j, v) in values.iter_mut().enumerate() {
                *v += self.weights[f * self.hidden + j];
            }
        }
        Accumulator { values, active }
    }
    pub fn update(&self, previous: &Accumulator, g: &Game) -> Accumulator {
        let active = self.active(g);
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
        let value = (self.output_bias
            + a.values
                .iter()
                .zip(&self.output)
                .map(|(x, w)| x.clamp(0.0, 1.0) * w)
                .sum::<f32>())
        .tanh();
        if value.is_finite() { value } else { 0.0 }
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
            && (1..=128).contains(&self.input)
            && (1..=128).contains(&self.output)
            && self.weights.len() == self.input * self.output * self.kernel * self.kernel
            && self.bias.len() == self.output
            && self.weights.iter().chain(&self.bias).all(|v| v.is_finite())
    }
    fn run(&self, input: &[f32], cancelled: &impl Fn() -> bool) -> Option<Vec<f32>> {
        let mut out = vec![0.0; self.output * SQ];
        if self.kernel == 1 {
            for oc in 0..self.output {
                let weights = &self.weights[oc * self.input..(oc + 1) * self.input];
                for y in 0..11 {
                    if cancelled() {
                        return None;
                    }
                    for x in 0..11 {
                        let square = y * 11 + x;
                        let mut value = self.bias[oc];
                        for (ic, weight) in weights.iter().enumerate() {
                            value += input[ic * SQ + square] * weight;
                        }
                        out[oc * SQ + square] = value;
                    }
                }
            }
            return Some(out);
        }
        let pad = (self.kernel / 2) as i32;
        for oc in 0..self.output {
            for y in 0..11 {
                if cancelled() {
                    return None;
                }
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
        Some(out)
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
        self.infer_cancellable(g, &|| false).unwrap()
    }
    pub fn infer_cancellable(
        &self,
        g: &Game,
        cancelled: &impl Fn() -> bool,
    ) -> Option<(f32, Vec<f32>)> {
        if cancelled() {
            return None;
        }
        let mut x = self.stem.run(&planes(g), cancelled)?;
        for v in &mut x {
            *v = v.max(0.0);
        }
        for block in &self.blocks {
            let mut y = block[0].run(&x, cancelled)?;
            for v in &mut y {
                *v = v.max(0.0);
            }
            let y = block[1].run(&y, cancelled)?;
            for (a, b) in x.iter_mut().zip(y) {
                *a = (*a + b).max(0.0);
            }
        }
        let v = self.value_bias
            + (0..self.channels)
                .map(|c| x[c * SQ..(c + 1) * SQ].iter().sum::<f32>() / SQ as f32 * self.value[c])
                .sum::<f32>();
        let value = v.tanh();
        let mut policy = self.policy.run(&x, cancelled)?;
        for logit in &mut policy {
            if !logit.is_finite() {
                *logit = 0.0;
            }
        }
        Some((if value.is_finite() { value } else { 0.0 }, policy))
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
    /// Explicit 0..100 percent override; None selects the bot's default.
    pub handcrafted_weight: Option<u8>,
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

#[cfg(test)]
mod convolution_tests {
    use super::*;

    #[test]
    fn pointwise_convolution_preserves_scalar_order_and_cancellation() {
        for channels in [1, 8, 17, 128] {
            let conv = Conv {
                input: channels,
                output: 40,
                kernel: 1,
                weights: (0..channels * 40)
                    .map(|i| (i as f32 % 19.0 - 9.0) / 13.0)
                    .collect(),
                bias: (0..40).map(|i| (i as f32 - 20.0) / 17.0).collect(),
            };
            let input: Vec<f32> = (0..channels * SQ)
                .map(|i| (i as f32 % 31.0 - 15.0) / 11.0)
                .collect();
            let actual = conv.run(&input, &|| false).unwrap();
            for oc in 0..40 {
                for square in 0..SQ {
                    let mut expected = conv.bias[oc];
                    for ic in 0..channels {
                        expected += input[ic * SQ + square] * conv.weights[oc * channels + ic];
                    }
                    assert_eq!(actual[oc * SQ + square].to_bits(), expected.to_bits());
                }
            }
            assert!(conv.run(&input, &|| true).is_none());
        }
    }
}
