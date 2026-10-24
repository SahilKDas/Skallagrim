# Uncommitted handcrafted-evaluation experiment

This working-tree experiment adds `handcrafted-v1` to Iota, Beta and Rho.
The score is always from the attacker's perspective and lies in [-0.95, 0.95]
for legal nonterminal positions. Authoritative terminal results remain exactly
-1, 0 or +1. Rho retains its CNN policy, and Beta retains uniform priors.

The positional terms are material balance, king distance to a corner, adjacent
attackers, nearby defender support, king sliding mobility, unobstructed corner
escape routes and sandwich-capture exposure. These are approximate positional
features, not tactical proofs. The coefficients in `src/evaluation.rs` are an
untuned initial hypothesis.

## Selecting a blend

The protocol option `HandcraftedWeight` accepts an integer percentage 0..100,
or -1 to select the bot's default:

```text
setoption name HandcraftedWeight value 25
```

25 means 75% neural value plus 25% handcrafted value. Zero preserves the neural
control and skips the new evaluator in the search hot path. Following the paired
comparison, Iota defaults to 25%; Beta and Rho default to 0%. The protocol default
is -1 (automatic), so switching bots selects the appropriate default. An explicit
percentage stays in effect across bot changes; -1 restores automatic selection.
`eval` reports the raw neural,
handcrafted and blended values. Baseline retains its historical evaluator.
The GUI accepts `--handcrafted-weight 25` for all three neural bots.

## Alternate campaign authorized October 21, 2026

`python/alternate_campaign.py` freezes the working binary, referee, source,
Python tooling and earned champion weights in a separate run. It does not commit,
push, replace deployed models or modify the original campaign.

The configured run is `runs/alternate-2000-2026-10-21`. Its source is the completed
`runs/league-5000-2026-10-13/revision-0002` campaign and the independently pinned
champions in `runs/search-speed-after-league/baseline.json`.

Stages run sequentially at Idle priority on logical CPU 6:

1. Verify Python/Rust neural inference parity.
2. Collect 1,000 additional completed role-balanced NNUE games, then warm-start
   its candidate on the full old corpus plus that extension.
3. Repeat with 1,000 additional CNN games. The total extension is 2,000 games.
4. Verify both candidate exports against Python inference.
5. For each of Iota, Beta and Rho, run paired 50-game comparisons at 300 ms per
   move with the existing 25-ms IPC allowance: 25% blend versus 0% using identical
   frozen champion weights. Separately compare the newly trained candidate at
   0% against the original champion at 0%. Replay-audit all six comparisons.

Collection uses 100-ms requests and a maximum of 600 plies, matching the earlier
campaign's collection settings; comparison clocks remain 300+25 ms. Failed,
held-out-contaminated and truncated games do not count toward the new corpus.
Collection uses the 0% control so changes to training and evaluation can be
measured independently. New seeds are 2026102101 and 2026102102.

Every old training/validation assignment is preserved. Of each 1,000-game
extension, 800 games train and 200 validate, balanced by candidate role. Each
combined corpus therefore has 5,800 training and 1,450 validation games.
Training uses ten CPU epochs with durable epoch recovery and a 0.0003 learning
rate. Best validation checkpoints are retained only in the alternate run.

The run has a 48-hour wall bound, 10,000 attempts per collection and a 2.8-GB
additional-storage cap, with a 2-GiB free-disk reserve. Peak additional storage
is estimated below 2 GB. No training or gauntlet stages run concurrently.
Committed collection transactions and epochs survive interruptions; restart the
same command to resume. A file lock prevents duplicate controllers. Host sleep
pauses work; the controller does not change system sleep settings.

```powershell
.venv/Scripts/python.exe python/alternate_campaign.py `
  --run runs/alternate-2000-2026-10-21 `
  --source-campaign runs/league-5000-2026-10-13/revision-0002 `
  --baseline runs/search-speed-after-league/baseline.json
```

`status.json`, collection progress, stage logs, replay audits, per-role scores,
paired bootstrap intervals and per-comparison `assessment.json` files are saved
inside the run. `results.json` appears after completion. Passing the screening
requires a paired confidence lower bound above 0.5, no infrastructure forfeits,
and at least 0.5 score in each role. Fifty games are a screening experiment;
small differences do not establish a durable strength gain. No model or blend
is automatically promoted.
