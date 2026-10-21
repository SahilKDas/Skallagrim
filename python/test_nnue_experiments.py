import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from dataset import DatasetIndex
import train

from models import NNUE, encode_nnue, migrate_nnue
from nnue_experiments import (
    augment_row,
    sample_key,
    target_for,
    training_offsets,
    transform_fen,
    validate_labels,
)
from referee import Referee
from transport import Engine
from nnue_dataset_report import report
from run_nnue_experiments import study_state

FEN = "3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a"


class ExperimentTests(unittest.TestCase):
    def test_search_telemetry_keeps_engine_and_transport_time_separate(self):
        engine = Engine.__new__(Engine)
        engine.send = Mock()
        engine.until = Mock(
            return_value=(
                "e1e2",
                ["info depth 4 score cp 12 nodes 900 time 280 pv e1e2"],
            )
        )
        move, elapsed, _, _ = engine.go(300, 25)
        self.assertEqual(move, "e1e2")
        self.assertEqual(engine.last_search_timing["engine_ms"], 280)
        self.assertEqual(engine.last_search_timing["nodes"], 900)
        self.assertEqual(engine.last_search_timing["depth"], 4)
        self.assertEqual(engine.last_search_timing["transport_ms"], elapsed)

    def test_study_resume_never_renews_hour_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "study.json"
            initial = study_state(path, {"seed": 42}, 100)
            resumed = study_state(path, {"seed": 42}, 500)
            self.assertEqual(initial["deadline"], 3700)
            self.assertEqual(resumed["deadline"], 3700)
            with self.assertRaises(ValueError):
                study_state(path, {"seed": 43}, 500)

    def test_coverage_report_preserves_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.jsonl"
            split = root / "split.json"
            data.write_text(
                "".join(
                    json.dumps(dict(game=g, fen=FEN, history=[FEN], target=1, ply=0))
                    + "\n"
                    for g in ["a", "b"]
                )
            )
            split.write_text(
                json.dumps(
                    dict(
                        data_sha256=DatasetIndex(data).sha256,
                        training=["a"],
                        validation=["b"],
                    )
                )
            )
            result = report(data, split)
            self.assertEqual(result["games"], 2)
            self.assertEqual(
                result["roles"], {"training/unknown": 1, "validation/unknown": 1}
            )
            split.write_text(
                json.dumps(
                    dict(
                        data_sha256=DatasetIndex(data).sha256,
                        training=["a", "b"],
                        validation=["b"],
                    )
                )
            )
            with self.assertRaises(ValueError):
                report(data, split)

    @unittest.skipUnless(
        os.environ.get("SKALLAGRIM_TEST_ENGINE"),
        "explicit experimental engine required",
    )
    def test_python_rust_experimental_feature_and_value_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = NNUE(64, "v2").eval()
            path = Path(temporary) / "nnue.json"
            model.export(path)
            with Engine(
                command=[os.environ["SKALLAGRIM_TEST_ENGINE"]],
                options={"Bot": "Iota", "NNUEFile": str(path)},
            ) as engine:
                rng = random.Random(16)
                for _ in range(40):
                    state = engine.snapshot()
                    if state["outcome"]:
                        engine.position([])
                        state = engine.snapshot()
                    f = encode_nnue(state["fen"], state["history"], "v2")
                    result = engine.evaluate()
                    self.assertEqual(
                        set(result["nnue_features"]),
                        set(f.nonzero().flatten().tolist()),
                    )
                    with torch.no_grad():
                        self.assertAlmostEqual(
                            model(f).item(), result["nnue"], places=5
                        )
                    engine.play(rng.choice(state["moves"]))

    def test_epoch_resume_matches_uninterrupted_and_early_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.jsonl"
            data.write_text(
                "".join(
                    json.dumps(dict(game=g, fen=FEN, history=[FEN], target=t)) + "\n"
                    for g, t in [("a", 1), ("b", -1), ("c", 1)]
                )
            )
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    dict(
                        data_sha256=DatasetIndex(data).sha256,
                        training=["a", "b"],
                        validation=["c"],
                    )
                )
            )

            def run(folder, extra=(), interrupted=False):
                folder.mkdir(exist_ok=True)
                args = [
                    "train.py",
                    "--data",
                    str(data),
                    "--kind",
                    "nnue",
                    "--output",
                    str(folder / "nnue.json"),
                    "--epoch-state",
                    str(folder / "epoch.pt"),
                    "--split",
                    str(split),
                    "--smoke",
                    "--batch",
                    "2",
                    "--epochs",
                    "3",
                    "--device",
                    "cpu",
                    *extra,
                ]
                if interrupted:
                    args += ["--deadline-unix", "10"]
                    # Interrupt validation of epoch two, after its updates but
                    # before its durable transaction can be committed.
                    clock = iter([0, 0, 0, 0, 0, 0, 0, 11])
                    with (
                        patch.object(sys, "argv", args),
                        patch.object(
                            train.time, "time", side_effect=lambda: next(clock)
                        ),
                    ):
                        with self.assertRaises(SystemExit) as stopped:
                            train.main()
                        self.assertEqual(stopped.exception.code, 75)
                else:
                    with patch.object(sys, "argv", args):
                        train.main()

            run(root / "full")
            run(root / "resume", interrupted=True)
            state = torch.load(root / "resume/epoch.pt", weights_only=True)
            self.assertEqual(state["epoch"], 1)
            run(root / "resume")
            full = torch.load(root / "full/epoch.pt", weights_only=True)
            resumed = torch.load(root / "resume/epoch.pt", weights_only=True)
            for key in full["net"]:
                self.assertTrue(torch.equal(full["net"][key], resumed["net"][key]))
            run(root / "early", ["--patience", "1", "--min-improvement", "100"])
            progress = json.loads((root / "early/nnue.progress.json").read_text())
            self.assertEqual(progress["epoch"], 2)

    def test_symmetries_keep_pieces_and_history_repetition(self):
        row = dict(game="g", fen=FEN, history=[FEN, FEN], target=1)
        for symmetry in range(8):
            changed = augment_row(row, symmetry)
            f = encode_nnue(changed["fen"], changed["history"], "v2")
            self.assertEqual(f[:363].sum(), 37)
            self.assertEqual(f[366], 1)
            self.assertEqual(changed["target"], 1)
        rotated = FEN
        for _ in range(4):
            rotated = transform_fen(rotated, 1)
        self.assertEqual(rotated, FEN)

    def test_expansion_preserves_evaluation_and_new_units_learn(self):
        torch.manual_seed(1)
        original = NNUE()
        for version, hidden in [("v1", 128), ("v2", 64)]:
            expanded = NNUE(hidden, version)
            migrate_nnue(expanded, copy.deepcopy(original.state_dict()))
            a = original(encode_nnue(FEN)[None])
            b = expanded(encode_nnue(FEN, [FEN], version)[None])
            self.assertTrue(torch.allclose(a, b, atol=1e-7))
            (b - 1).square().backward()
            if hidden > 64:
                self.assertGreater(expanded.output.weight.grad[0, 64:].abs().sum(), 0)

    def test_sampling_excludes_validation(self):
        index = Mock(groups={"train": [1, 2, 3], "valid": [4, 5]})
        self.assertEqual(
            len(training_offsets(index, {"valid"}, 2, random.Random(1))), 2
        )
        self.assertNotIn(4, training_offsets(index, {"valid"}, 0, random.Random(1)))

    def test_labels_require_training_provenance_and_exact_sample(self):
        row = dict(game="g", fen=FEN, history=[FEN], target=1)
        manifest = dict(
            data_sha256="data",
            perspective="Attacker",
            training_only=True,
            engine_sha256="a" * 64,
            model_sha256="b" * 64,
            labels={sample_key(row): dict(game="g", value=-1)},
        )
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            artifact.write_bytes(b"fixture")
            for kind in ["engine", "model"]:
                manifest[kind + "_path"] = str(artifact)
                manifest[kind + "_sha256"] = hashlib.sha256(b"fixture").hexdigest()
            labels = validate_labels(manifest, "data", {"g"})
            self.assertEqual(target_for(row, labels, 0.5), 0)
            with self.assertRaises(ValueError):
                validate_labels(manifest, "data", {"other"})
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_labels(manifest, "data", {"g"})
        with self.assertRaises(ValueError):
            target_for(row, {}, 0.5)

    def test_referee_failure_poisons_connection(self):
        ref = Referee.__new__(Referee)
        ref.failed = False
        ref.sequence = 0
        ref.state = {}
        ref.latencies = []
        ref.proc = Mock()
        ref.proc.pid = 1
        ref.replies = Mock()
        ref.replies.get.side_effect = queue.Empty
        with tempfile.TemporaryDirectory() as temporary:
            ref.log_dir = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                ref.call("clock")
            ref.proc.stdin.write.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "unusable"):
                ref.call("snapshot")
            ref.proc.stdin.write.assert_not_called()
            self.assertTrue((ref.log_dir / "protocol-failure-1.json").exists())

    def test_deadline_before_initialization_does_not_commit_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nnue.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("train.py")),
                    "--data",
                    str(Path(temporary) / "missing.sqlite"),
                    "--kind",
                    "nnue",
                    "--output",
                    str(output),
                    "--deadline-unix",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse((Path(temporary) / "epoch-state.pt").exists())
            self.assertFalse(
                json.loads(output.with_suffix(".budget.json").read_text())["complete"]
            )


if __name__ == "__main__":
    unittest.main()
