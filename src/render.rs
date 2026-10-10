//! All board, chrome, pieces and text rasterization is performed with tiny-skia.
use crate::game::{Game, Piece, Side, corner};
use fontdue::Font;
use tiny_skia::*;
pub struct Renderer {
    font: Option<Font>,
}
#[derive(Clone, Copy)]
pub struct Layout {
    pub x: f32,
    pub y: f32,
    pub cell: f32,
    pub panel: f32,
}
impl Layout {
    pub fn new(w: u32, h: u32) -> Self {
        let board = (h as f32 - 148.0).min(w as f32 - 330.0).max(110.0);
        Self {
            x: 48.0,
            y: 98.0,
            cell: board / 11.0,
            panel: board + 82.0,
        }
    }
    pub fn square(&self, x: f32, y: f32) -> Option<usize> {
        let col = ((x - self.x) / self.cell).floor() as i32;
        let row = ((y - self.y) / self.cell).floor() as i32;
        if (0..11).contains(&col) && (0..11).contains(&row) {
            Some((10 - row) as usize * 11 + col as usize)
        } else {
            None
        }
    }
    pub fn center(&self, s: usize) -> (f32, f32) {
        (
            self.x + (s % 11) as f32 * self.cell + self.cell / 2.0,
            self.y + (10 - s / 11) as f32 * self.cell + self.cell / 2.0,
        )
    }
}
pub struct View<'a> {
    pub game: &'a Game,
    pub selected: Option<usize>,
    pub hover: Option<usize>,
    pub cursor: usize,
    pub bot: &'a str,
    pub mode: &'a str,
    pub status: &'a str,
    pub moves: &'a [String],
    pub thinking: bool,
    pub mouse: Option<(f32, f32)>,
    pub rules_open: bool,
}
fn paint(rgb: (u8, u8, u8)) -> Paint<'static> {
    let mut p = Paint::default();
    p.set_color_rgba8(rgb.0, rgb.1, rgb.2, 255);
    p.anti_alias = true;
    p
}
fn rect(pix: &mut Pixmap, x: f32, y: f32, w: f32, h: f32, rgb: (u8, u8, u8)) {
    if let Some(r) = Rect::from_xywh(x, y, w, h) {
        pix.fill_rect(r, &paint(rgb), Transform::identity(), None);
    }
}
fn circle(pix: &mut Pixmap, x: f32, y: f32, radius: f32, rgb: (u8, u8, u8)) {
    if let Some(p) = PathBuilder::from_circle(x, y, radius) {
        pix.fill_path(
            &p,
            &paint(rgb),
            FillRule::Winding,
            Transform::identity(),
            None,
        );
    }
}
fn line(pix: &mut Pixmap, a: (f32, f32), b: (f32, f32), width: f32, color: (u8, u8, u8)) {
    let mut path = PathBuilder::new();
    path.move_to(a.0, a.1);
    path.line_to(b.0, b.1);
    pix.stroke_path(
        &path.finish().unwrap(),
        &paint(color),
        &Stroke {
            width,
            line_cap: LineCap::Round,
            ..Stroke::default()
        },
        Transform::identity(),
        None,
    );
}
impl Default for Renderer {
    fn default() -> Self {
        let candidates = [
            "assets/ui.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ];
        let font = candidates.iter().find_map(|p| {
            std::fs::read(p)
                .ok()
                .and_then(|b| Font::from_bytes(b, fontdue::FontSettings::default()).ok())
        });
        Self { font }
    }
}
impl Renderer {
    pub fn text(
        &self,
        pix: &mut Pixmap,
        text: &str,
        x: f32,
        baseline: f32,
        size: f32,
        rgb: (u8, u8, u8),
    ) {
        let Some(font) = &self.font else {
            return;
        };
        let mut x = x;
        for c in text.chars() {
            let (m, bitmap) = font.rasterize(c, size);
            if m.width > 0 && m.height > 0 {
                let mut glyph = Pixmap::new(m.width as u32, m.height as u32).unwrap();
                for (dst, &a) in glyph
                    .data_mut()
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(&bitmap)
                {
                    dst[0] = (rgb.0 as u16 * a as u16 / 255) as u8;
                    dst[1] = (rgb.1 as u16 * a as u16 / 255) as u8;
                    dst[2] = (rgb.2 as u16 * a as u16 / 255) as u8;
                    dst[3] = a;
                }
                pix.draw_pixmap(
                    x.round() as i32 + m.xmin,
                    baseline.round() as i32 - m.height as i32 - m.ymin,
                    glyph.as_ref(),
                    &PixmapPaint::default(),
                    Transform::identity(),
                    None,
                );
            }
            x += m.advance_width;
        }
    }
    pub fn draw(&self, w: u32, h: u32, v: View<'_>) -> Pixmap {
        let mut pix = Pixmap::new(w, h).unwrap();
        pix.fill(Color::from_rgba8(18, 25, 29, 255));
        let l = Layout::new(w, h);
        let cream = (229, 226, 211);
        let muted = (143, 160, 161);
        let gold = (222, 177, 92);
        self.text(&mut pix, "SKALLAGRIM", 48.0, 43.0, 28.0, cream);
        self.text(&mut pix, "COPENHAGEN / 11 × 11", 49.0, 66.0, 12.0, muted);
        let board = l.cell * 11.0;
        rect(
            &mut pix,
            l.x - 5.0,
            l.y - 5.0,
            board + 10.0,
            board + 10.0,
            (48, 60, 62),
        );
        let destinations: Vec<_> = v
            .game
            .legal_moves()
            .into_iter()
            .filter(|m| Some(m.from) == v.selected)
            .map(|m| m.to)
            .collect();
        for s in 0..121 {
            let (cx, cy) = l.center(s);
            let (x, y) = (cx - l.cell / 2.0, cy - l.cell / 2.0);
            let special = corner(s) || s == 60;
            let color = if Some(s) == v.selected {
                (133, 127, 81)
            } else if v.game.last_move.is_some_and(|m| m.from == s || m.to == s) {
                (112, 128, 106)
            } else if special {
                (111, 105, 84)
            } else if (s % 11 + s / 11) % 2 == 0 {
                (192, 180, 151)
            } else {
                (181, 168, 139)
            };
            rect(&mut pix, x, y, l.cell - 0.6, l.cell - 0.6, color);
            if special {
                let r = l.cell * 0.23;
                line(&mut pix, (cx - r, cy), (cx, cy - r), 1.5, (155, 142, 106));
                line(&mut pix, (cx, cy - r), (cx + r, cy), 1.5, (155, 142, 106));
                line(&mut pix, (cx + r, cy), (cx, cy + r), 1.5, (155, 142, 106));
                line(&mut pix, (cx, cy + r), (cx - r, cy), 1.5, (155, 142, 106));
            }
            if destinations.contains(&s) {
                circle(&mut pix, cx, cy, l.cell * 0.09, (45, 106, 92));
            }
            if v.game.captures.contains(&s) {
                let r = l.cell * 0.16;
                line(
                    &mut pix,
                    (cx - r, cy - r),
                    (cx + r, cy + r),
                    2.5,
                    (161, 66, 53),
                );
                line(
                    &mut pix,
                    (cx - r, cy + r),
                    (cx + r, cy - r),
                    2.5,
                    (161, 66, 53),
                );
            }
            let p = v.game.board[s];
            if p != Piece::Empty {
                let radius = l.cell * 0.34;
                circle(&mut pix, cx + 1.0, cy + 3.0, radius + 1.0, (100, 92, 77));
                let outer = if p == Piece::Attacker {
                    (25, 40, 44)
                } else if p == Piece::King {
                    gold
                } else {
                    (240, 232, 204)
                };
                circle(&mut pix, cx, cy, radius, outer);
                circle(
                    &mut pix,
                    cx,
                    cy,
                    radius * 0.77,
                    if p == Piece::Attacker {
                        (43, 60, 63)
                    } else if p == Piece::King {
                        (244, 204, 124)
                    } else {
                        (218, 207, 175)
                    },
                );
                if p == Piece::King {
                    let r = radius * 0.5;
                    let mut path = PathBuilder::new();
                    path.move_to(cx - r, cy + r * 0.5);
                    path.line_to(cx - r, cy - r * 0.65);
                    path.line_to(cx - r * 0.45, cy - r * 0.1);
                    path.line_to(cx, cy - r);
                    path.line_to(cx + r * 0.45, cy - r * 0.1);
                    path.line_to(cx + r, cy - r * 0.65);
                    path.line_to(cx + r, cy + r * 0.5);
                    path.close();
                    pix.fill_path(
                        &path.finish().unwrap(),
                        &paint((93, 63, 31)),
                        FillRule::Winding,
                        Transform::identity(),
                        None,
                    );
                } else {
                    let r = radius * 0.35;
                    line(
                        &mut pix,
                        (cx - r, cy),
                        (cx + r, cy),
                        1.6,
                        if p == Piece::Attacker {
                            (139, 155, 151)
                        } else {
                            (134, 122, 97)
                        },
                    );
                    if p == Piece::Attacker {
                        line(&mut pix, (cx, cy - r), (cx, cy + r), 1.6, (139, 155, 151));
                    }
                }
            }
            if Some(s) == v.hover || s == v.cursor {
                let color = if s == v.cursor { (69, 119, 115) } else { gold };
                let b = 2.0;
                rect(&mut pix, x + b, y + b, l.cell - 2.0 * b, 1.5, color);
                rect(
                    &mut pix,
                    x + b,
                    y + l.cell - b - 1.5,
                    l.cell - 2.0 * b,
                    1.5,
                    color,
                );
                rect(&mut pix, x + b, y + b, 1.5, l.cell - 2.0 * b, color);
                rect(
                    &mut pix,
                    x + l.cell - b - 1.5,
                    y + b,
                    1.5,
                    l.cell - 2.0 * b,
                    color,
                );
            }
        }
        for i in 0..11 {
            self.text(
                &mut pix,
                &((b'a' + i as u8) as char).to_string(),
                l.x + (i as f32 + 0.43) * l.cell,
                l.y + board + 22.0,
                12.0,
                muted,
            );
            self.text(
                &mut pix,
                &(11 - i).to_string(),
                l.x - 27.0,
                l.y + (i as f32 + 0.57) * l.cell,
                12.0,
                muted,
            );
        }
        let px = l.panel;
        let pw = (w as f32 - px - 28.0).max(220.0);
        rect(&mut pix, px, 98.0, pw, h as f32 - 146.0, (25, 34, 39));
        let turn = if v.game.side == Side::Attacker {
            "ATTACKERS"
        } else {
            "DEFENDERS"
        };
        self.text(
            &mut pix,
            if v.game.outcome.is_some() {
                "GAME COMPLETE"
            } else {
                turn
            },
            px + 20.0,
            132.0,
            20.0,
            cream,
        );
        self.text(
            &mut pix,
            if v.thinking {
                "Engine is thinking…"
            } else if v.game.outcome.is_some() {
                "Start a new game or view replay"
            } else {
                "Ready to play"
            },
            px + 20.0,
            157.0,
            13.0,
            muted,
        );
        self.text(
            &mut pix,
            &format!("Engine: {}", v.bot),
            px + 20.0,
            193.0,
            16.0,
            gold,
        );
        self.text(&mut pix, v.mode, px + 20.0, 217.0, 13.0, muted);
        let buttons = [
            ("NEW GAME   [N]", 242.0),
            ("UNDO   [U]", 286.0),
            ("ENGINE MOVE   [SPACE]", 330.0),
            ("CHANGE BOT   [B]", 374.0),
            ("CHANGE SIDE   [H]", 418.0),
            ("SAVE REPLAY   [S]", 462.0),
        ];
        for (label, y) in buttons {
            let hovered = v.mouse.is_some_and(|(x, my)| {
                x >= px + 16.0 && x <= px + pw - 16.0 && my >= y && my < y + 34.0
            });
            rect(
                &mut pix,
                px + 16.0,
                y,
                pw - 32.0,
                34.0,
                if hovered { (52, 78, 79) } else { (37, 50, 55) },
            );
            self.text(&mut pix, label, px + 28.0, y + 22.0, 12.0, cream);
        }
        self.text(&mut pix, "MOVE HISTORY", px + 20.0, 533.0, 12.0, muted);
        let rows = ((h as f32 - 635.0) / 23.0).max(0.0) as usize;
        for (i, m) in v
            .moves
            .iter()
            .enumerate()
            .skip(v.moves.len().saturating_sub(rows))
        {
            let row = i - v.moves.len().saturating_sub(rows);
            self.text(
                &mut pix,
                &format!("{:>3}.  {}", i + 1, m),
                px + 20.0,
                558.0 + row as f32 * 23.0,
                14.0,
                cream,
            );
        }
        self.text(&mut pix, v.status, 48.0, h as f32 - 19.0, 12.0, muted);
        if let Some(o) = &v.game.outcome {
            self.text(
                &mut pix,
                &format!(
                    "{} / {}",
                    match o.winner {
                        Some(Side::Attacker) => "Attackers win",
                        Some(Side::Defender) => "Defenders win",
                        None => "Draw",
                    },
                    o.reason
                ),
                48.0,
                84.0,
                12.0,
                gold,
            );
        }
        if v.rules_open {
            rect(
                &mut pix,
                l.x + 16.0,
                l.y + 16.0,
                board - 32.0,
                board - 32.0,
                (25, 34, 39),
            );
            self.text(
                &mut pix,
                "COPENHAGEN RULES",
                l.x + 38.0,
                l.y + 60.0,
                22.0,
                gold,
            );
            let lines = [
                "Attackers move first. Every piece moves like a rook.",
                "Only the king can land on the throne or corners.",
                "Everyone can cross the empty throne. The king may return.",
                "Capture a soldier by closing an orthogonal sandwich.",
                "The king is armed; restricted squares can act as capturers.",
                "Shieldwalls need edge brackets, inward support and a flanking move.",
                "Shieldwalls capture soldiers beside the king, but spare the king.",
                "Defenders win by corner escape or an unbreakable, mobile exit fort.",
                "Capture the king on four sides; the adjacent throne replaces one.",
                "The king cannot be captured on an edge. Full encirclement wins.",
                "Third board-and-turn repetition loses for defenders. No moves loses.",
                "An impossible-to-finish draw requires referee adjudication [D].",
                "Additional arbiter-certified exit forts: [F]. See docs/RULES.md.",
                "Press R to return to play.",
            ];
            for (i, text) in lines.iter().enumerate() {
                self.text(
                    &mut pix,
                    text,
                    l.x + 38.0,
                    l.y + 100.0 + i as f32 * 29.0,
                    12.0,
                    cream,
                );
            }
        }
        pix
    }
}
