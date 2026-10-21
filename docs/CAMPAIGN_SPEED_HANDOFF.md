# Campaign completion and search optimization handoff

Authorized by the user on October 17, 2026. Continue the existing frozen campaign
before running speed experiments. This document does not alter its manifest.

## Finish existing work

Complete NNUE promotion, the single CNN round (6,250 accepted audited games;
5,000 training / 1,250 validation; exact role split; ten warm-start epochs), and
the 750-game comparison suite and replay audits. Preserve checkpoints, saved
games, original schedules, gates, search clocks, resource ceilings and approved
recovery overlays. Never run duplicate controllers or repeat completed epochs.
The latest controller identity and recovery are recorded in the run, not here.
Report checkpoint selection, role scores, paired confidence intervals, exclusions,
timeouts and upstream limitations. Rejected candidates remain experimental.

## Pin the actual champions

After final completion and publication, run:

```powershell
.venv/Scripts/python.exe python/pin_campaign_baseline.py --run runs/league-5000-2026-10-13/revision-0002 --output runs/search-speed-after-league/baseline.json
```

The helper independently rechecks promotion decisions, frozen input hashes and
model hashes. It selects each accepted candidate or the frozen incumbent when
promotion failed. Candidate-suite results are not silently relabelled as champion
results. The baseline contains the engine configuration, model and binary hashes,
decision and audit hashes, and fixed 300-ms / 25-ms clocks. Keep its weights fixed.
Close the campaign monitor after final reporting and the verified baseline handoff.

## Optimize outside the campaign

Use the pinned baseline, CPU 6 and Idle priority. First record a deterministic,
role-balanced fixed-position benchmark with completed alpha-beta depth, nodes,
MCTS simulations, inference time, total response latency and fallback counts.
Use the frozen evaluation positions, preserving full histories and testing both
sides; distinguish terminal positions from search trials. Add telemetry if absent
and measure its overhead before comparing engines. Never equate MCTS depth with
completed alpha-beta depth.

Profile to identify the largest bottleneck. Change one search/evaluation hot path
at a time in the working source; never edit the frozen runtime or baseline weights.
Verify rules, legal moves and neural parity. Require timing probes and a two-hour
soak with original clock limits, then paired 50-game gauntlets and replay audits
under the existing role and confidence gates before adopting an optimized engine.
Speed metrics alone do not establish strength. No additional training rounds,
relaxed clocks, changed resource ceilings or automatic promotion are authorized.

The later October 17 authorization in [NNUE_IMPROVEMENTS.md](NNUE_IMPROVEMENTS.md)
adds a single post-campaign, one-hour training experiment on existing data. It
does not authorize another collection round or modify the frozen campaign.
