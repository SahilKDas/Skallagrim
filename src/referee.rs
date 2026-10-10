//! Authoritative match state, clocks and auditable adjudication.
use crate::game::{Game, Move, Outcome, RULES_VERSION, START};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

pub struct Referee {
    pub game: Game,
    pub revision: u64,
    states: Vec<Game>,
    moves: Vec<String>,
    events: Vec<Value>,
    start: String,
    clock: Option<(Instant, u64)>,
    pub analysis: Value,
}
impl Default for Referee {
    fn default() -> Self {
        let game = Game::default();
        Self {
            states: vec![game.clone()],
            game,
            revision: 0,
            moves: vec![],
            events: vec![],
            start: START.into(),
            clock: None,
            analysis: Value::Null,
        }
    }
}
impl Referee {
    fn log(&mut self, kind: &str, evidence: Value) {
        let reason = evidence.get("reason").cloned().unwrap_or_else(|| {
            json!(match kind {
                "move" => "Legal move; Copenhagen captures and repetition transition",
                "analysis" => "Bounded certificate analysis",
                "start" => "Validated starting position and opening",
                "rewind" => "Human requested rewind",
                "clock-start" => "Turn deadline armed",
                _ => kind,
            })
        });
        self.events.push(json!({"sequence":self.events.len(),"kind":kind,"reason":reason,"revision":self.revision,"ply":self.moves.len(),"fen":self.game.fen(),"rules":RULES_VERSION,"evidence":evidence,"outcome":self.game.outcome}));
    }
    pub fn tick(&mut self, now: Instant) {
        if self
            .clock
            .is_some_and(|(start, ms)| now.duration_since(start) >= Duration::from_millis(ms))
            && self.game.outcome.is_none()
        {
            let (start, ms) = self.clock.take().unwrap();
            self.game.outcome = Some(Outcome {
                winner: Some(self.game.side.other()),
                reason: "timeout".into(),
            });
            self.revision += 1;
            self.log("ruling", json!({"source":"clock","reason":"timeout","budget_ms":ms,"elapsed_ms":now.duration_since(start).as_millis()}));
        }
    }
    pub fn snapshot(&self) -> Value {
        let mut state = self.game.snapshot();
        state["board"] = json!(self.game.board.to_vec());
        state["last_move"] = json!(self.game.last_move);
        state["revision"] = json!(self.revision);
        state["analysis"] = self.analysis.clone();
        state["last_ruling"] = self
            .events
            .iter()
            .rev()
            .find(|e| e["kind"] == "ruling")
            .cloned()
            .unwrap_or(Value::Null);
        state["clock"] = self.clock.map(|(start, budget)| json!({"budget_ms":budget,"remaining_ms":budget.saturating_sub(start.elapsed().as_millis() as u64)})).unwrap_or(Value::Null);
        state
    }
    pub fn export(&self) -> Value {
        json!({"format":"skallagrim-referee-replay-v1","start":self.start,"moves":self.moves,"final_fen":self.game.fen(),"outcome":self.game.outcome,"referee":{"version":2,"semantics":"dynamic-fort-v1","rules":RULES_VERSION,"events":self.events}})
    }
    pub fn evidence(&mut self, revision: u64, result: Value) {
        if revision == self.revision {
            self.analysis = result;
            self.log("analysis", self.analysis.clone());
        }
    }
    pub fn verified_fort(&mut self, revision: u64, proof: crate::fort::VerifiedFort) {
        if revision != self.revision
            || self.game.outcome.is_some()
            || !proof.matches(&self.game)
            || self.analysis["claim"] != "fort"
            || self.analysis["status"] != "proven"
        {
            return;
        }
        self.game.outcome = Some(Outcome {
            winner: Some(crate::game::Side::Defender),
            reason: "exit-fort-dynamic".into(),
        });
        self.clock = None;
        self.revision += 1;
        self.log("ruling", json!({"source":"fort-certificate","reason":"exit-fort-dynamic","analysis":self.analysis,"verified":true}));
    }
    pub fn command(&mut self, request: &Value, now: Instant) -> Result<Value, String> {
        self.tick(now);
        let command = request["command"].as_str().ok_or("missing command")?;
        if request.get("revision").is_some_and(|v| !v.is_u64()) {
            return Err("invalid position revision".into());
        }
        if let Some(expected) = request["revision"].as_u64()
            && expected != self.revision
        {
            return Err("stale position revision".into());
        }
        match command {
            "snapshot" => return Ok(self.snapshot()),
            "export" => return Ok(self.export()),
            "validate" => {
                if self.clock.is_some() {
                    return Err("pause the clock before replay validation".into());
                }
                let replay = crate::replay::Replay::from_value(&request["replay"])?;
                return Ok(json!({"valid":true,"plies":replay.moves.len(),"rules":RULES_VERSION}));
            }
            "new" => {
                let start = match request.get("fen") {
                    Some(value) => value.as_str().ok_or("invalid starting FEN")?,
                    None => START,
                };
                if request.get("moves").is_some_and(|v| !v.is_array()) {
                    return Err("invalid opening moves".into());
                }
                let mut next = Game::from_fen(start)?;
                let mut states = vec![next.clone()];
                let mut moves = vec![];
                if let Some(items) = request["moves"].as_array() {
                    for item in items {
                        let text = item.as_str().ok_or("invalid opening move")?;
                        next.play(text.parse()?)?;
                        states.push(next.clone());
                        moves.push(text.to_string());
                    }
                }
                self.game = next;
                self.states = states;
                self.moves = moves;
                self.start = start.into();
                self.events.clear();
                self.clock = None;
                self.analysis = Value::Null;
                self.revision += 1;
                self.log("start", json!({"start":self.start,"moves":self.moves}));
                if let Some(outcome) = &self.game.outcome {
                    self.log("ruling",json!({"source":"rules","reason":outcome.reason,"proof":"Validated starting board and opening moves"}));
                }
            }
            "clock" => {
                if self.game.outcome.is_some() || self.clock.is_some() {
                    return Err("clock already active or game finished".into());
                }
                let ms = request["milliseconds"]
                    .as_u64()
                    .filter(|ms| *ms > 0 && *ms <= 86_400_000)
                    .ok_or("invalid clock budget")?;
                self.clock = Some((now, ms));
                self.log("clock-start", json!({"budget_ms":ms,"side":self.game.side}));
            }
            "pause" => {
                if self.clock.is_some()
                    && request["reason"]
                        .as_str()
                        .is_none_or(|v| v.trim().is_empty())
                {
                    return Err("pause requires a reason".into());
                }
                if self.clock.take().is_some() {
                    self.log(
                        "clock-pause",
                        json!({"source":"human","reason":request["reason"]}),
                    );
                }
            }
            "play" => {
                let text = request["move"].as_str().ok_or("missing move")?;
                let mv: Move = text.parse()?;
                self.game.play(mv)?;
                let elapsed = self
                    .clock
                    .take()
                    .map(|(start, _)| now.duration_since(start).as_millis());
                self.moves.push(text.into());
                self.states.push(self.game.clone());
                self.analysis = Value::Null;
                self.revision += 1;
                self.log(
                    "move",
                    json!({"move":text,"captures":self.game.captures,"elapsed_ms":elapsed}),
                );
                if let Some(outcome) = &self.game.outcome {
                    self.log("ruling", json!({"source":"rules","reason":outcome.reason,"proof":"Copenhagen deterministic board and repetition rules"}));
                }
            }
            "undo" => {
                let count = request["count"].as_u64().unwrap_or(1) as usize;
                if count == 0 || count > self.moves.len() {
                    return Err("invalid undo count".into());
                }
                self.moves.truncate(self.moves.len() - count);
                self.states.truncate(self.states.len() - count);
                self.game = self.states.last().unwrap().clone();
                self.clock = None;
                self.analysis = Value::Null;
                self.revision += 1;
                self.log("rewind", json!({"count":count,"source":"human"}));
            }
            "adjudicate" => {
                let operator = request["operator"]
                    .as_str()
                    .filter(|v| !v.trim().is_empty())
                    .ok_or("human identity required")?;
                let note = request["reason"]
                    .as_str()
                    .filter(|v| !v.trim().is_empty())
                    .ok_or("human decision reason required")?;
                let claim = request["claim"].as_str().ok_or("missing claim")?;
                self.game.referee_decision(claim)?;
                self.clock = None;
                self.revision += 1;
                self.log("ruling", json!({"source":"human","operator":operator,"reason":note,"claim":claim,"analysis":self.analysis}));
            }
            "forfeit" | "limit" => {
                if self.game.outcome.is_some() {
                    return Ok(self.snapshot());
                }
                let reason = if command == "limit" {
                    "move-limit-adjudication"
                } else {
                    request["reason"].as_str().ok_or("missing forfeit reason")?
                };
                if command == "forfeit" && reason == "move-limit-adjudication" {
                    return Err("use limit for a move-limit draw".into());
                }
                if ![
                    "move-limit-adjudication",
                    "timeout",
                    "engine-failure",
                    "illegal-move",
                    "rules-incompatibility",
                ]
                .contains(&reason)
                {
                    return Err("invalid administrative result".into());
                }
                self.game.outcome = Some(Outcome {
                    winner: if command == "limit" {
                        None
                    } else {
                        Some(self.game.side.other())
                    },
                    reason: reason.into(),
                });
                self.clock = None;
                self.revision += 1;
                self.log("ruling", json!({"source":"match-controller","reason":reason,"evidence":request["evidence"]}));
            }
            _ => return Err("unknown referee command".into()),
        }
        Ok(self.snapshot())
    }
}
