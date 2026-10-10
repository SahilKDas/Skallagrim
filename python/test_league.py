"""League policy, isolation and promotion checks without training a network."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from league import (
    make_schedule,
    load_positions,
    passes,
    gate_accepted,
    manifest,
    verify_manifest,
    pin,
    run_gates,
)
from dataset import DatasetIndex
from records import save
from transport import ROOT


def summary(attacker=0.6, defender=0.6):
    return dict(
        games=50,
        completed=True,
        paired_bootstrap_95=[0.51, 0.7],
        termination_counts={},
        by_side={
            "Attacker": dict(games=25, score_rate=attacker),
            "Defender": dict(games=25, score_rate=defender),
        },
    )


class LeagueTests(unittest.TestCase):
    def test_deterministic_schedule_and_paired_roles(self):
        current = {"bot": "Beta", "model": "current"}
        champion = {"bot": "Beta", "model": "champion"}
        history = [{"bot": "Beta", "model": "old"}]
        positions = [{"id": "hard"}]
        rows = make_schedule(current, champion, history, positions, 10000, 42)
        self.assertEqual(
            rows, make_schedule(current, champion, history, positions, 10000, 42)
        )
        for left, right in zip(rows[::2], rows[1::2]):
            self.assertEqual(left["opponent"], right["opponent"])
            self.assertEqual(left["position"], right["position"])
            self.assertEqual(
                (left["candidate_side"], right["candidate_side"]),
                ("Attacker", "Defender"),
            )
        self.assertEqual(sum(row["position"] is not None for row in rows), 2000)
        fractions = {
            kind: sum(row["category"] == kind for row in rows) / len(rows)
            for kind in ["champion", "historical", "self"]
        }
        for kind, expected in [("champion", 0.5), ("historical", 0.3), ("self", 0.2)]:
            self.assertLess(abs(fractions[kind] - expected), 0.03)
        self.assertTrue(
            all(
                row["category"] == "self"
                for row in make_schedule(current, None, [], positions, 10, 1)
            )
        )

    def test_position_identity_and_partition_isolation(self):
        path = ROOT / "benchmarks/league-positions.json"
        value = load_positions(path)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "positions.json"
            changed = copy.deepcopy(value)
            changed["evaluation"][0] = changed["training"][0]
            save(target, changed)
            with self.assertRaises(ValueError):
                load_positions(target)
            changed = copy.deepcopy(value)
            changed["training"][0]["moves"].append("a1a2")
            save(target, changed)
            with self.assertRaises(ValueError):
                load_positions(target)

    def test_whole_game_removed_on_held_out_visit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "samples.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        dict(game="a", target=1, fen="ordinary a"),
                        dict(game="a", target=1, fen="held-out d"),
                        dict(game="b", target=-1, fen="ordinary d"),
                    ]
                )
            )
            self.assertEqual(set(DatasetIndex(path, {"held-out d"}).groups), {"b"})

    def test_promotion_requires_each_role_opponent_and_nnue_search(self):
        self.assertTrue(passes(summary(), True))
        self.assertFalse(passes(summary(0.9, 0.4)))
        bad = summary()
        bad["termination_counts"]["timeout"] = 1
        self.assertFalse(passes(bad))
        bad = summary()
        bad["completed"] = False
        self.assertFalse(passes(bad))
        bad = summary()
        bad["paired_bootstrap_95"][0] = 0.5
        self.assertFalse(passes(bad, True))
        gates = {
            "iota/champion": summary(),
            "beta/champion": summary(),
            "iota/held-out": summary(),
            "beta/held-out": summary(),
        }
        self.assertTrue(gate_accepted(gates, {"iota/champion", "beta/champion"}))
        gates["beta/held-out"] = summary(0.8, 0.3)
        self.assertFalse(gate_accepted(gates, {"iota/champion", "beta/champion"}))

    def test_manifest_resume_artifacts_schedule_and_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "league.json"
            suite = ROOT / "benchmarks/league-positions.json"
            current = dict(bot="Baseline", model=None)
            value, _ = manifest(path, current, None, [], suite, 2, 20, 42, 30, 600)
            verify_manifest(value)
            manifest(path, current, None, [], suite, 2, 20, 42, 30, 600)
            with self.assertRaises(ValueError):
                manifest(path, current, None, [], suite, 2, 20, 43, 30, 600)
            changed = copy.deepcopy(value)
            changed["schedule"][0]["candidate_side"] = "Defender"
            with self.assertRaises(ValueError):
                verify_manifest(changed)
            changed = copy.deepcopy(value)
            changed["engine"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                verify_manifest(changed)

    def test_nnue_runs_both_searches_and_historical_and_hard_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model = ROOT / "models/nnue.json"
            calls = []

            def fake_run(*args, budget=None):
                calls.append(args)
                output = Path(args[args.index("--output") + 1])
                output.mkdir(parents=True, exist_ok=True)
                start = args.index("--opponents") + 1
                end = args.index("--output")
                enemies = list(args[start:end])
                enemies = (
                    enemies[: enemies.index("--openings")]
                    if "--openings" in enemies
                    else enemies
                )
                for enemy in enemies:
                    save(output / f"{enemy}-summary.json", summary())

            accepted, summaries = run_gates(
                folder,
                "nnue",
                model,
                model,
                [model],
                ROOT / "benchmarks/league-positions.json",
                fake_run,
                None,
            )
            self.assertTrue(accepted)
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(summaries), 6)
            self.assertEqual(pin(model)["sha256"], pin(model)["sha256"])


if __name__ == "__main__":
    unittest.main()
