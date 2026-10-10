//! Validate replay moves, starting positions and recorded terminal decisions.
use crate::game::{Game, Move, Outcome, START};
use serde_json::Value;

pub struct Replay {
    pub states: Vec<Game>,
    pub moves: Vec<String>,
}
impl Replay {
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let start = match value.get("start") {
            Some(start) => start.as_str().ok_or("invalid replay start")?,
            None => START,
        };
        let mut game = Game::from_fen(start)?;
        let mut states = vec![game.clone()];
        let mut moves = vec![];
        let opening = value.get("opening").and_then(|v| v.get("moves"));
        if opening.is_some_and(|v| !v.is_array()) {
            return Err("invalid opening moves".into());
        }
        let played = value
            .get("moves")
            .and_then(Value::as_array)
            .ok_or("replay requires moves")?;
        for item in opening
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .chain(played)
        {
            if let Some(side) = item.get("side") {
                let recorded =
                    serde_json::from_value(side.clone()).map_err(|_| "invalid replay side")?;
                if game.side != recorded {
                    return Err("replay side mismatch".into());
                }
            }
            let text = item
                .as_str()
                .or_else(|| item.get("move").and_then(Value::as_str))
                .ok_or("invalid replay move")?;
            game.play(text.parse::<Move>()?)?;
            moves.push(text.to_string());
            states.push(game.clone());
        }
        if let Some(fen) = value.get("final_fen")
            && fen.as_str() != Some(game.fen().as_str())
        {
            return Err("replay final board mismatch".into());
        }
        let recorded: Option<Option<Outcome>> = if let Some(outcome) = value.get("outcome") {
            Some(serde_json::from_value(outcome.clone()).map_err(|_| "invalid replay outcome")?)
        } else if let Some(reason) = value.get("reason") {
            Some(Some(Outcome {
                winner: serde_json::from_value(
                    value
                        .get("winner")
                        .cloned()
                        .ok_or("missing replay winner")?,
                )
                .map_err(|_| "invalid replay winner")?,
                reason: reason.as_str().ok_or("invalid replay reason")?.to_string(),
            }))
        } else {
            None
        };
        if let Some(recorded) = recorded {
            if game.outcome.is_some() {
                if game.outcome != recorded {
                    return Err("replay terminal outcome mismatch".into());
                }
            } else if let Some(outcome) = recorded {
                match outcome.reason.as_str() {
                    "exit-fort-dynamic"
                        if outcome.winner == Some(crate::game::Side::Defender)
                            && value["referee"]["version"] == 2 =>
                    {
                        game.outcome = Some(outcome.clone());
                    }
                    "exit-fort-referee" => game.referee_decision("fort")?,
                    "impossible-to-finish-referee" => game.referee_decision("draw")?,
                    "timeout" | "engine-failure" | "illegal-move" | "rules-incompatibility" => {
                        if outcome.winner != Some(game.side.other()) {
                            return Err("replay forfeit winner mismatch".into());
                        }
                        game.outcome = Some(outcome.clone());
                    }
                    "move-limit-adjudication" if outcome.winner.is_none() => {
                        game.outcome = Some(outcome.clone())
                    }
                    _ => {
                        return Err(
                            "recorded outcome has no supporting move or referee decision".into(),
                        );
                    }
                }
                if game.outcome.as_ref() != Some(&outcome) {
                    return Err("replay referee winner mismatch".into());
                }
            }
            *states.last_mut().unwrap() = game;
        }
        if value.get("referee").is_some() {
            crate::referee_audit::validate(value, &moves, states.last().unwrap())?;
        }
        Ok(Self { states, moves })
    }
    pub fn to_value(&self) -> Value {
        serde_json::json!({"format":"skallagrim-replay-v1", "start":self.states[0].fen(),
            "moves":self.moves, "outcome":self.states.last().unwrap().outcome})
    }
}
