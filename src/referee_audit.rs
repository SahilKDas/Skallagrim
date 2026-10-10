//! Validate the referee event timeline, including branches revoked by undo.
use crate::game::{Game, Move, Outcome, RULES_VERSION};
use serde_json::Value;

fn verify_analysis(game: &Game, evidence: &Value) -> Result<(), String> {
    match evidence["claim"].as_str() {
        Some("fort") => {
            if !["proven", "unknown"].contains(&evidence["status"].as_str().unwrap_or(""))
                || (evidence["status"] == "proven"
                    && !game.exit_fort()
                    && evidence["proof"]["certificate"].is_null())
            {
                return Err("unsupported fort certificate".into());
            }
            if !evidence["proof"]["certificate"].is_null() {
                let cert = serde_json::from_value::<crate::fort::Certificate>(
                    evidence["proof"]["certificate"].clone(),
                )
                .map_err(|e| e.to_string())?;
                if evidence["status"] != "proven" {
                    return Err("certificate attached to unknown result".into());
                }
                crate::fort::verify(game, &cert)?;
            }
        }
        Some("draw") => {
            let proof = &evidence["proof"];
            let expected = match proof["status"].as_str() {
                Some("Unknown") => "unknown",
                Some("FinishReachable") => "finish-reachable",
                Some("ProvenDraw") => "proven",
                _ => return Err("missing draw proof".into()),
            };
            if evidence["status"] != expected
                || proof["reason"].as_str().is_none_or(|v| v.trim().is_empty())
            {
                return Err("inconsistent draw evidence".into());
            }
            let positions = proof["positions"]
                .as_u64()
                .filter(|v| *v > 0 && *v <= 100000)
                .ok_or("invalid proof position count")? as usize;
            let witness = proof["witness"].as_array().ok_or("missing proof witness")?;
            if expected == "finish-reachable" {
                let mut finish = game.clone();
                finish.history = crate::game::History::default();
                finish.history.push(finish.fen().into());
                finish.adjudicate();
                for text in witness {
                    finish.history = crate::game::History::default();
                    finish.history.push(finish.fen().into());
                    finish.play(text.as_str().ok_or("invalid proof move")?.parse::<Move>()?)?;
                }
                if finish.outcome.is_none()
                    || (proof["reason"] != "already terminal"
                        && proof["reason"].as_str()
                            != finish.outcome.as_ref().map(|o| o.reason.as_str()))
                {
                    return Err("witness does not prove a decisive finish".into());
                }
            } else if !witness.is_empty() {
                return Err("unexpected draw witness".into());
            }
            if expected == "proven" {
                let verified = crate::proof::prove_draw(
                    game,
                    positions,
                    std::time::Duration::from_secs(10),
                    &std::sync::atomic::AtomicBool::new(false),
                );
                if verified.status != crate::proof::ProofStatus::ProvenDraw
                    || verified.positions != positions
                {
                    return Err("closed draw graph could not be reverified".into());
                }
            }
        }
        _ => return Err("invalid analysis claim".into()),
    }
    Ok(())
}

pub fn validate(value: &Value, expected_moves: &[String], expected: &Game) -> Result<(), String> {
    let audit = &value["referee"];
    if ![Some(1), Some(2)].contains(&audit["version"].as_u64())
        || audit["rules"] != RULES_VERSION
        || (audit["version"] == 2 && audit["semantics"] != "dynamic-fort-v1")
    {
        return Err("unsupported referee audit version or rules".into());
    }
    let mut game = Game::from_fen(value["start"].as_str().unwrap_or(crate::game::START))?;
    let mut states = vec![game.clone()];
    let mut moves = Vec::<String>::new();
    let events = audit["events"].as_array().ok_or("missing referee events")?;
    let mut clock = None;
    let mut revision = 0;
    let mut analysis = Value::Null;
    for (index, event) in events.iter().enumerate() {
        if event["sequence"].as_u64() != Some(index as u64) || event["rules"] != RULES_VERSION {
            return Err("invalid referee event sequence or rules".into());
        }
        let next_revision = event["revision"]
            .as_u64()
            .ok_or("missing referee revision")?;
        if next_revision < revision {
            return Err("referee revision went backwards".into());
        }
        revision = next_revision;
        let evidence = &event["evidence"];
        match event["kind"].as_str().ok_or("missing referee event kind")? {
            "start" => {
                if index != 0 || evidence["start"].as_str() != Some(game.fen().as_str()) {
                    return Err("invalid referee start event".into());
                }
                for item in evidence["moves"]
                    .as_array()
                    .ok_or("missing initial moves")?
                {
                    let text = item.as_str().ok_or("invalid initial move")?;
                    game.play(text.parse::<Move>()?)?;
                    moves.push(text.into());
                    states.push(game.clone());
                }
            }
            "move" => {
                analysis = Value::Null;
                let text = evidence["move"].as_str().ok_or("missing audited move")?;
                game.play(text.parse::<Move>()?)?;
                if evidence["captures"] != serde_json::json!(game.captures) {
                    return Err("audited captures mismatch".into());
                }
                if let Some(budget) = clock
                    && evidence["elapsed_ms"]
                        .as_u64()
                        .is_none_or(|elapsed| elapsed >= budget)
                {
                    return Err("move has missing or expired clock evidence".into());
                }
                clock = None;
                moves.push(text.into());
                states.push(game.clone());
            }
            "rewind" => {
                analysis = Value::Null;
                let count = evidence["count"].as_u64().ok_or("invalid rewind")? as usize;
                if count == 0 || count > moves.len() {
                    return Err("invalid audited rewind".into());
                }
                moves.truncate(moves.len() - count);
                states.truncate(states.len() - count);
                game = states.last().unwrap().clone();
                clock = None;
            }
            "clock-start" => {
                if clock.is_some()
                    || game.outcome.is_some()
                    || evidence["side"] != serde_json::json!(game.side)
                {
                    return Err("invalid clock start".into());
                }
                clock = Some(
                    evidence["budget_ms"]
                        .as_u64()
                        .filter(|v| *v > 0)
                        .ok_or("invalid audited clock budget")?,
                );
            }
            "clock-pause" => {
                if clock.take().is_none()
                    || (evidence["source"] != "human"
                        || evidence["reason"]
                            .as_str()
                            .is_none_or(|v| v.trim().is_empty()))
                {
                    return Err("invalid clock pause".into());
                }
            }
            "analysis" => {
                if evidence["revision"].as_u64() != Some(revision) {
                    return Err("analysis position mismatch".into());
                }
                verify_analysis(&game, evidence)?;
                analysis = evidence.clone();
            }
            "ruling" => {
                match evidence["source"].as_str().ok_or("ruling has no source")? {
                    "fort-certificate" => {
                        if audit["version"] != 2
                            || game.outcome.is_some()
                            || evidence["analysis"] != analysis
                            || analysis["claim"] != "fort"
                            || analysis["status"] != "proven"
                            || evidence["verified"] != true
                        {
                            return Err("invalid dynamic fort ruling".into());
                        }
                        let cert = serde_json::from_value::<crate::fort::Certificate>(
                            analysis["proof"]["certificate"].clone(),
                        )
                        .map_err(|e| e.to_string())?;
                        crate::fort::verify(&game, &cert)?;
                        game.outcome = Some(Outcome {
                            winner: Some(crate::game::Side::Defender),
                            reason: "exit-fort-dynamic".into(),
                        });
                    }
                    "rules" => {
                        if game
                            .outcome
                            .as_ref()
                            .is_none_or(|o| Some(o.reason.as_str()) != evidence["reason"].as_str())
                        {
                            return Err("unproved automatic ruling".into());
                        }
                    }
                    "human" => {
                        if evidence["analysis"] != analysis {
                            return Err("human ruling analysis mismatch".into());
                        }
                        if evidence["operator"]
                            .as_str()
                            .is_none_or(|s| s.trim().is_empty())
                            || evidence["reason"]
                                .as_str()
                                .is_none_or(|s| s.trim().is_empty())
                        {
                            return Err("human ruling lacks identity or reason".into());
                        }
                        game.referee_decision(
                            evidence["claim"].as_str().ok_or("missing human claim")?,
                        )?;
                    }
                    "clock" => {
                        let budget = clock.ok_or("timeout without active clock")?;
                        if game.outcome.is_some()
                            || evidence["budget_ms"].as_u64() != Some(budget)
                            || evidence["elapsed_ms"].as_u64().is_none_or(|ms| ms < budget)
                        {
                            return Err("unproved timeout".into());
                        }
                        game.outcome = Some(Outcome {
                            winner: Some(game.side.other()),
                            reason: "timeout".into(),
                        });
                    }
                    "match-controller" => {
                        if game.outcome.is_some() {
                            return Err("ruling overwrites existing result".into());
                        }
                        let reason = evidence["reason"]
                            .as_str()
                            .ok_or("missing administrative reason")?;
                        if ![
                            "move-limit-adjudication",
                            "timeout",
                            "engine-failure",
                            "illegal-move",
                            "rules-incompatibility",
                        ]
                        .contains(&reason)
                        {
                            return Err("invalid administrative ruling".into());
                        }
                        game.outcome = Some(Outcome {
                            winner: if reason == "move-limit-adjudication" {
                                None
                            } else {
                                Some(game.side.other())
                            },
                            reason: reason.into(),
                        });
                    }
                    _ => return Err("unknown ruling source".into()),
                }
                clock = None;
            }
            _ => return Err("unknown referee event".into()),
        }
        let outcome: Option<Outcome> = serde_json::from_value(event["outcome"].clone())
            .map_err(|_| "invalid audited outcome")?;
        if event["fen"].as_str() != Some(game.fen().as_str())
            || event["ply"].as_u64() != Some(moves.len() as u64)
            || game.outcome != outcome
        {
            return Err("referee timeline position or outcome mismatch".into());
        }
    }
    if moves != expected_moves || game.fen() != expected.fen() || game.outcome != expected.outcome {
        return Err("referee audit does not support final replay".into());
    }
    Ok(())
}
