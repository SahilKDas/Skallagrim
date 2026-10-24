//! Experimental positional scores from the attacker's perspective.
use crate::game::{DIRS, Game, Piece, SQ, Side, THRONE, corner, step};

pub const VERSION: &str = "handcrafted-v1";

pub fn evaluate(game: &Game) -> f32 {
    if let Some(value) = crate::search::terminal(game) {
        return value;
    }
    let Some(king) = game.king() else { return 1.0 };
    let mut attackers = 0;
    let mut defenders = 0;
    let mut vulnerable_attackers = 0;
    let mut vulnerable_defenders = 0;
    let mut adjacent_attackers = 0;
    let mut king_support = 0;
    for square in 0..SQ {
        let piece = game.board[square];
        match piece {
            Piece::Attacker => attackers += 1,
            Piece::Defender => defenders += 1,
            _ => continue,
        }
        let distance = (square % 11).abs_diff(king % 11) + (square / 11).abs_diff(king / 11);
        if piece == Piece::Attacker && distance == 1 {
            adjacent_attackers += 1;
        }
        if piece == Piece::Defender && distance <= 2 {
            king_support += 1;
        }
        // Enemy/anvil on one side and a vacant capture square on the other.
        // A positional risk proxy: the opponent may not reach that square.
        let victim = piece.side().unwrap();
        let exposed = DIRS.iter().any(|&(dx, dy)| {
            let (Some(a), Some(b)) = (step(square, dx, dy), step(square, -dx, -dy)) else {
                return false;
            };
            let hostile = game.board[a].side() == Some(victim.other())
                || corner(a)
                || (a == THRONE && (victim == Side::Attacker || game.board[a] == Piece::Empty));
            hostile && game.board[b] == Piece::Empty && !corner(b) && b != THRONE
        });
        if exposed {
            if piece == Piece::Attacker {
                vulnerable_attackers += 1;
            } else {
                vulnerable_defenders += 1;
            }
        }
    }
    let (x, y) = (king % 11, king / 11);
    let corner_distance = x.min(10 - x) + y.min(10 - y);
    let mut king_mobility = 0;
    let mut escape_routes = 0;
    for (dx, dy) in DIRS {
        let mut at = king;
        while let Some(next) = step(at, dx, dy) {
            if game.board[next] != Piece::Empty {
                break;
            }
            king_mobility += 1;
            if corner(next) {
                escape_routes += 1;
            }
            at = next;
        }
    }
    let raw = 0.045 * (attackers as f32 - 2.0 * defenders as f32)
        + 0.055 * (corner_distance as f32 - 6.0)
        + 0.10 * adjacent_attackers as f32
        - 0.025 * king_support as f32
        - 0.012 * king_mobility as f32
        - (if game.side == Side::Defender {
            0.8
        } else {
            0.4
        }) * escape_routes as f32
        - 0.05 * vulnerable_attackers as f32
        + 0.08 * vulnerable_defenders as f32;
    raw.tanh().clamp(-0.95, 0.95)
}

pub fn blend(game: &Game, neural: f32, percent: u8) -> f32 {
    if let Some(value) = crate::search::terminal(game) {
        return value;
    }
    if percent == 0 && neural.is_finite() {
        return neural;
    }
    let handcrafted = evaluate(game);
    if !neural.is_finite() {
        return handcrafted;
    }
    let weight = percent.min(100) as f32 / 100.0;
    ((1.0 - weight) * neural + weight * handcrafted).clamp(-1.0, 1.0)
}
