"""Streaming, whole-game budget and optimization-equivalence regressions."""

import hashlib
import json
import random
import subprocess
import sys
import os
import time
import tempfile
import unittest
from pathlib import Path
from dataset import DatasetIndex, open_data, archive_data, make_replay
from transport import ROOT
from resources import StorageBudget
from pipeline import run

FEN = "3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a"


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")


class DatasetTests(unittest.TestCase):
    def test_storage_limit_stops_owned_producer_and_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = root / "producer.py"
            script.write_text(
                "import subprocess,sys,time,pathlib\nroot=pathlib.Path(sys.argv[1])\nchild=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n(root/'child.pid').write_text(str(child.pid))\ntry:\n (root/'data.bin').write_bytes(b'x'*16384)\n time.sleep(60)\nfinally:\n child.terminate()\n child.wait(timeout=5)\n",
                encoding="utf8",
            )
            started = time.perf_counter()
            with self.assertRaisesRegex(RuntimeError, "storage budget"):
                run(script, root, budget=StorageBudget(root, 4096, reserve_bytes=0))
            self.assertLess(time.perf_counter() - started, 12)
            pid = int((root / "child.pid").read_text())
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [
                    wintypes.DWORD,
                    wintypes.BOOL,
                    wintypes.DWORD,
                ]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x100000, False, pid)
                if handle:
                    try:
                        self.assertEqual(kernel.WaitForSingleObject(handle, 0), 0)
                    finally:
                        kernel.CloseHandle(handle)
            else:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_interleaved_games_and_gzip_preserve_offsets_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.jsonl"
            rows = [
                {"game": game, "fen": FEN[:-1] + side, "target": target}
                for game, side, target in [
                    ("α", "a", 1),
                    ("b", "d", -1),
                    ("α", "d", 1),
                    ("ignored", "a", None),
                ]
            ]
            write_rows(path, rows)
            original = path.read_bytes()
            index = DatasetIndex(path)
            self.assertEqual(index.sides["α"], {"a": 1, "d": 1})
            self.assertNotIn("ignored", index.groups)
            offsets = sorted(
                offset for group in index.groups.values() for offset in group
            )
            with open_data(path) as stream:
                self.assertEqual(DatasetIndex.read_batch(stream, offsets), rows[:3])
            archive_data(path)
            self.assertFalse(path.exists())
            zipped = path.with_suffix(".jsonl.gz")
            index = DatasetIndex(zipped)
            self.assertEqual(index.sha256, hashlib.sha256(original).hexdigest())
            with open_data(zipped) as stream:
                self.assertEqual(DatasetIndex.read_batch(stream, offsets), rows[:3])

    def test_replay_keeps_recent_whole_games_and_enforces_floor(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b, out = [
                Path(folder) / name for name in ["a.jsonl", "b.jsonl", "replay.jsonl"]
            ]
            rows = [
                {"game": game, "fen": FEN, "target": 1, "ply": ply}
                for game in range(6)
                for ply in range(2)
            ]
            write_rows(a, rows[:6])
            write_rows(b, rows[6:])
            archive_data(b)
            result = make_replay(
                [a, b.with_suffix(".jsonl.gz")], out, 3, 100000, minimum_games=3
            )
            self.assertEqual(result["games"], 3)
            actual = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(actual, rows[6:])
            before = out.read_bytes()
            with self.assertRaises(ValueError):
                make_replay([a], out, 3, 1, minimum_games=3)
            self.assertEqual(out.read_bytes(), before)

    def test_streamed_training_matches_eager_optimization(self):
        import torch
        from models import NNUE, encode

        torch.set_num_threads(1)
        # Synthetic smoke fixtures test optimization equivalence, not strength.
        rows = [
            {
                "game": str(game),
                "fen": FEN[:-1] + side,
                "history": [FEN],
                "legal": [],
                "policy": [],
                "target": float(1 if game % 2 else -1),
            }
            for game in range(5)
            for side in ["a", "d"]
        ]
        with tempfile.TemporaryDirectory() as folder:
            data, output = Path(folder) / "data.jsonl", Path(folder) / "nnue.json"
            write_rows(data, rows)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "python/train.py"),
                    "--data",
                    str(data),
                    "--kind",
                    "nnue",
                    "--output",
                    str(output),
                    "--epochs",
                    "1",
                    "--batch",
                    "3",
                    "--hidden",
                    "4",
                    "--seed",
                    "11",
                    "--device",
                    "cpu",
                    "--smoke",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                timeout=120,
            )
            torch.manual_seed(11)
            random.seed(11)
            games = sorted({row["game"] for row in rows})
            random.shuffle(games)
            valid = set(games[:1])
            training = [row for row in rows if row["game"] not in valid]
            random.shuffle(training)
            net = NNUE(4)
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
            for start in range(0, len(training), 3):
                batch = training[start : start + 3]
                x = torch.stack(
                    [encode(row["fen"], row["history"])[1] for row in batch]
                )
                target = torch.tensor([row["target"] for row in batch])
                optimizer.zero_grad()
                loss = (net(x) - target).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()
            actual = torch.load(output.with_suffix(".pt"), weights_only=True)
            for key, expected in net.state_dict().items():
                self.assertTrue(torch.equal(actual[key], expected), key)
            meta = json.loads(output.with_suffix(".meta.json").read_text())
            self.assertEqual((meta["training_games"], meta["validation_games"]), (4, 1))
            self.assertTrue(meta["smoke"])


if __name__ == "__main__":
    unittest.main()
