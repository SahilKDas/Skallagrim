"""Campaign transaction, exact-count, recovery and orchestration regressions."""

import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from campaign_state import Collection, atomic_gzip, role_split
from dataset import DatasetIndex, make_replay_sqlite, open_data
from league import pin
from pipeline import validate_checkpoint
from records import save
from resources import StorageBudget
from run_campaign import freeze, normalized_engines
from transport import ROOT

FEN = "3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a"


def row(game, role):
    return dict(
        game=str(game),
        fen=FEN if role == "Attacker" else FEN[:-1] + "d",
        history=[],
        target=1.0,
        legal=["d1d2" if role == "Attacker" else "e5e3"],
        policy=[["d1d2" if role == "Attacker" else "e5e3", 1.0]],
        league=dict(candidate_side=role),
    )


def checkpoint(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    path.with_suffix(".pt").write_bytes(b"fixture")
    save(
        path.with_suffix(".meta.json"),
        dict(
            training_games=1000,
            validation_games=250,
            smoke=False,
            model_artifacts={
                p.name: pin(p)["sha256"] for p in (path, path.with_suffix(".pt"))
            },
        ),
    )


class CampaignTests(unittest.TestCase):
    def test_compressed_replay_preserves_every_row_and_random_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "samples.jsonl"
            source.write_text(
                "".join(
                    json.dumps(row(i, "Attacker" if i % 2 == 0 else "Defender")) + "\n"
                    for i in range(16)
                )
            )
            database = root / "replay.sqlite"
            selection = make_replay_sqlite([source], database, 16, 1024**2, 16)
            before, after = DatasetIndex(source), DatasetIndex(database)
            self.assertEqual(selection["games"], 16)
            self.assertEqual(before.sha256, after.sha256)
            self.assertEqual(before.roles, after.roles)
            with open_data(database) as stream:
                batch = DatasetIndex.read_batch(
                    stream, [after.groups["15"][0], after.groups["0"][0]]
                )
            self.assertEqual([item["game"] for item in batch], ["15", "0"])
            with self.assertRaises(ValueError):
                make_replay_sqlite([source], root / "too-small.sqlite", 16, 1, 16)
            self.assertTrue(source.exists())

    def test_first_seeded_candidate_is_gated_and_rejection_keeps_champion(self):
        import pipeline

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = root / "seed/nnue.json"
            checkpoint(seed)

            def stage(*args, **kwargs):
                if args[0] == "python/selfplay.py":
                    output = Path(args[args.index("--output") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(
                        "".join(
                            json.dumps(row(i, "Attacker" if i % 2 == 0 else "Defender"))
                            + "\n"
                            for i in range(1250)
                        )
                    )
                elif args[0] == "python/train.py":
                    output = Path(args[args.index("--output") + 1])
                    checkpoint(output)
                    meta = json.loads(output.with_suffix(".meta.json").read_text())
                    save(
                        output.with_suffix(".meta.json"),
                        meta | {"training_complete": True},
                    )

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "pipeline",
                        "--kind",
                        "nnue",
                        "--generations",
                        "1",
                        "--games",
                        "1250",
                        "--initial-champion",
                        str(seed),
                        "--run",
                        str(root / "run"),
                    ],
                ),
                patch.object(pipeline, "run", side_effect=stage),
                patch(
                    "league.run_gates", return_value=(False, {"reason": "failed roles"})
                ) as gate,
            ):
                pipeline.main()
            gate.assert_called_once()
            self.assertEqual(gate.call_args.args[3], seed.resolve())
            complete = json.loads(
                (root / "run/generation-0000/complete.json").read_text()
            )
            self.assertFalse(complete["accepted"])
            self.assertEqual(complete["promoted_model"], str(seed.resolve()))
            self.assertTrue((root / "run/generation-0000/nnue.json").exists())

    def test_transactions_resume_assemble_and_ignore_uncommitted_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = dict(games=2, manifest="pinned")
            collection = Collection(root / "transactions", config)
            collection.commit(
                0, "Attacker", [row(0, "Attacker")], {"game": "0"}, 1, 0, None
            )
            collection.commit(
                1, "Defender", [row(1, "Defender")], {"game": "1"}, 1, 1, None
            )
            (root / "transactions/attempt-000002.json.gz.tmp").write_bytes(
                b"interrupted"
            )
            seen = []
            resumed = Collection(root / "transactions", config, seen.append)
            self.assertEqual(resumed.completed, 2)
            self.assertEqual(resumed.roles, dict(Attacker=1, Defender=1))
            self.assertEqual(resumed.next_attempt, 2)
            self.assertEqual(len(seen), 2)
            resumed.assemble(root / "samples.gz", root / "audit.gz")
            self.assertEqual(len(DatasetIndex(root / "samples.gz").groups), 2)
            resumed.assemble(root / "samples.gz", root / "audit.gz")
            self.assertEqual(len(DatasetIndex(root / "samples.gz").groups), 2)
            with self.assertRaises(ValueError):
                Collection(root / "transactions", config | {"manifest": "changed"})
            with gzip.open(
                root / "transactions/attempt-000000.json.gz", "rt"
            ) as stream:
                record = json.load(stream)
            record["samples"][0]["target"] = -1
            atomic_gzip(root / "transactions/attempt-000000.json.gz", record)
            with self.assertRaises(ValueError):
                Collection(root / "transactions", config)

    def test_exact_role_split_and_changed_split_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data.jsonl"
            with data.open("w") as stream:
                for game in range(6250):
                    stream.write(
                        json.dumps(
                            row(game, "Attacker" if game % 2 == 0 else "Defender")
                        )
                        + "\n"
                    )
            index = DatasetIndex(data)
            split = role_split(index, root / "split.json", 5000, 1250, 42)
            self.assertEqual(len(split["training"]), 5000)
            self.assertEqual(len(split["validation"]), 1250)
            self.assertFalse(set(split["training"]) & set(split["validation"]))
            for role in ("Attacker", "Defender"):
                self.assertEqual(
                    sum(index.roles[g] == role for g in split["training"]), 2500
                )
                self.assertEqual(
                    sum(index.roles[g] == role for g in split["validation"]), 625
                )
            self.assertEqual(
                split, role_split(index, root / "split.json", 5000, 1250, 42)
            )
            with self.assertRaises(ValueError):
                role_split(index, root / "split.json", 5000, 1250, 43)

    def test_excluded_games_cannot_enter_split(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.jsonl"
            path.write_text(json.dumps(row(1, "Attacker")) + "\n")
            self.assertFalse(DatasetIndex(path, {FEN}).groups)
            collection = Collection(Path(temp) / "transactions", dict(games=2))
            value = row(1, "Attacker") | {"target": None}
            collection.commit(
                0, "Attacker", [value], {}, 1, 0, "infrastructure-failure"
            )
            self.assertEqual(collection.completed, 0)
            self.assertEqual(collection.exclusions["infrastructure-failure"], 1)

    def test_seed_pin_and_storage_count_floor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("skallagrim.exe", "skallagrim-referee.exe"):
                path = root / "workspace/target/release" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture executable")
            for kind in ("nnue", "cnn"):
                checkpoint(root / "workspace/models" / f"{kind}.json")
            python = root / "workspace/python"
            python.mkdir()
            (python / "run_campaign.py").write_bytes(
                (ROOT / "python/run_campaign.py").read_bytes()
            )
            bench = root / "workspace/benchmarks"
            bench.mkdir()
            for name in ("league-positions.json", "openings.json"):
                (bench / name).write_text("{}")
            save(bench / "engines.json", dict(engines={}))
            save(
                bench / "search-fix-verification.json",
                dict(
                    status="pass",
                    runtime_sha256={
                        name: pin(root / "workspace/target/release" / name)["sha256"]
                        for name in ("skallagrim.exe", "skallagrim-referee.exe")
                    },
                ),
            )
            runtime, _ = freeze(root / "run", root / "workspace")
            freeze(root / "run", root / "workspace")
            (runtime / "skallagrim.exe").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                freeze(root / "run", root / "workspace")
            with self.assertRaises(ValueError):
                validate_checkpoint(root / "workspace/models/nnue.json", 5000, 1250)
            with self.assertRaises(RuntimeError):
                StorageBudget(root, 0).check()

    def test_epoch_resume_and_configuration_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "samples.jsonl"
            data.write_text(
                "".join(
                    json.dumps(row(i, "Attacker" if i % 2 == 0 else "Defender")) + "\n"
                    for i in range(16)
                )
            )
            role_split(DatasetIndex(data), root / "split.json", 12, 4, 42)
            command = [
                sys.executable,
                str(ROOT / "python/train.py"),
                "--data",
                str(data),
                "--kind",
                "nnue",
                "--output",
                str(root / "nnue.json"),
                "--epochs",
                "2",
                "--batch",
                "4",
                "--device",
                "cpu",
                "--smoke",
                "--split",
                str(root / "split.json"),
                "--epoch-state",
                str(root / "epoch.pt"),
            ]
            for attempt in range(2):
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=45
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                if attempt == 0:
                    (root / "nnue.json").write_text("interrupted export")
            self.assertEqual(
                json.loads((root / "nnue.progress.json").read_text())["epoch"], 2
            )
            before = pin(root / "nnue.pt")
            changed = list(command)
            changed[changed.index("--epochs") + 1] = "3"
            result = subprocess.run(changed, capture_output=True, text=True, timeout=45)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Epoch resume rejected", result.stderr)
            self.assertEqual(pin(root / "nnue.pt"), before)
            command.remove("--smoke")
            command += ["--training-games", "5000", "--validation-games", "1250"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("5,000", result.stderr)

    def test_campaign_config_propagation_and_paths(self):
        import gauntlet_suite

        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "candidate.json"
            commands = []

            class Process:
                def wait(self):
                    return 0

                def poll(self):
                    return 0

            with (
                patch.object(
                    sys,
                    "argv",
                    ["gauntlet_suite", "--output", temp, "--config", str(config)],
                ),
                patch.object(
                    gauntlet_suite.subprocess,
                    "Popen",
                    side_effect=lambda command, **kw: (
                        commands.append(command) or Process()
                    ),
                ),
                patch.object(gauntlet_suite.subprocess, "run"),
            ):
                gauntlet_suite.main()
            self.assertEqual(len(commands), 3)
            self.assertTrue(
                all(
                    command[command.index("--config") + 1] == str(config)
                    for command in commands
                )
            )
            value = normalized_engines(
                dict(
                    engines={
                        "opentafl": dict(
                            command=["python", "./python/adapter.py", "./a;./b"],
                            artifacts=["x"],
                        )
                    }
                ),
                ROOT,
            )
            entry = value["engines"]["opentafl"]
            self.assertEqual(entry["command"][0], sys.executable)
            self.assertEqual(
                entry["command"][2], str(ROOT / "a") + ";" + str(ROOT / "b")
            )


if __name__ == "__main__":
    unittest.main()
