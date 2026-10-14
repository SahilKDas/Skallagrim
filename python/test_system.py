"""Integration checks for protocol atomicity, cancellation, neural parity gates."""

import json
import subprocess
import unittest
import tempfile
from pathlib import Path
from transport import Engine, ROOT
from benchmark import summarize, paired_interval


class SystemTests(unittest.TestCase):
    def test_report_uses_recorded_settings_and_requires_game_logs(self):
        import sys

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = dict(
                options=dict(
                    candidate="rho",
                    movetime=777,
                    games=50,
                    ipc_allowance=0,
                    max_plies=999,
                    cpu=2,
                ),
                strength_evidence=False,
                platform="test-platform",
                processor="test-processor",
                rules="test-rules",
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            summary = summarize(
                [
                    dict(
                        score=1,
                        candidate_side="Attacker" if i % 2 == 0 else "Defender",
                        reason="corner-escape",
                    )
                    for i in range(50)
                ],
                42,
            )
            summary["completed"] = True
            (root / "iota-summary.json").write_text(json.dumps(summary))
            output = root / "report.md"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "python/report.py"),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            text = output.read_text(encoding="utf8")
            self.assertIn("777 ms search, 0 ms IPC allowance, 999 plies", text)
            self.assertIn("test-processor / test-platform", text)
            self.assertNotIn("i7-1255U", text)
            self.assertEqual(
                json.loads(output.with_suffix(".json").read_text())[
                    "complete_pairings"
                ],
                0,
            )

    def test_protocol_rejects_startpos_junk(self):
        with Engine() as e:
            before = e.snapshot()["fen"]
            e.send("position startpos junk")
            e.send("isready")
            with self.assertRaises(RuntimeError):
                e.until("readyok")
            e.until("readyok")
            self.assertEqual(e.snapshot()["fen"], before)

    def test_draw_proof_cancels_without_changing_game(self):
        with Engine() as e:
            before = e.snapshot()["fen"]
            e.send("prove draw nodes 100000 movetime 60000")
            e.send("stop")
            result = json.loads(e.until("proof", 2)[0])
            self.assertEqual(result["status"], "Unknown")
            self.assertEqual(e.snapshot()["fen"], before)
            e.ready()

    def test_selfplay_invalid_limits_and_truncation_are_bounded(self):
        import sys

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "samples.jsonl"
            base = [
                sys.executable,
                str(ROOT / "python/selfplay.py"),
                "--output",
                str(output),
            ]
            invalid = subprocess.run(
                base + ["--max-plies", "0"], capture_output=True, text=True, timeout=5
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertFalse(output.exists())
            truncated = subprocess.run(
                base
                + [
                    "--games",
                    "1",
                    "--max-plies",
                    "1",
                    "--max-attempts",
                    "1",
                    "--movetime",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(truncated.returncode, 0)
            self.assertIn("partial corpus preserved", truncated.stderr)
            self.assertTrue(output.exists())

    def test_pipeline_rejects_incompatible_resume_before_launch(self):
        import sys

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = dict(
                kind="cnn",
                games=1250,
                movetime=100,
                epochs=10,
                window_games=20000,
                device="auto",
            )
            (root / "config.json").write_text(json.dumps(config))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "python/pipeline.py"),
                    "--kind",
                    "nnue",
                    "--run",
                    str(root),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Resume rejected: changed kind", result.stderr)
            self.assertEqual(json.loads((root / "config.json").read_text()), config)

    def test_protocol_invalid_position_is_atomic(self):
        with Engine() as e:
            before = e.snapshot()["fen"]
            e.send("position startpos moves a1b1")
            e.send("isready")
            with self.assertRaises(RuntimeError):
                e.until("readyok")
            e.until("readyok")
            self.assertEqual(e.snapshot()["fen"], before)

    def test_cancellation_has_one_bestmove_and_recovers(self):
        with Engine() as e:
            e.send("go infinite")
            e.send("stop")
            move, _ = e.until("bestmove", 2)
            self.assertIn(move, e.snapshot()["moves"])
            e.ready()
            move, *_ = e.go(20)
            self.assertIn(move, e.snapshot()["moves"])

    def test_neural_bots_require_weights(self):
        with Engine(options={"Bot": "Rho"}) as e:
            with self.assertRaises(RuntimeError):
                e.go(10)

    def test_neural_cancellation_has_one_reply_and_reuses_worker(self):
        for bot in ("Iota", "Beta", "Rho"):
            with self.subTest(bot=bot), Engine(options={
                "Bot": bot, "NNUEFile": str(ROOT / "models/nnue.json"),
                "CNNFile": str(ROOT / "models/cnn.json"),
            }) as engine:
                legal = engine.snapshot()["moves"]
                for _ in range(3):
                    engine.send("go infinite")
                    engine.send("stop")
                    move, _ = engine.until("bestmove", 2)
                    self.assertIn(move, legal)
                    engine.ready()
                    move, _, _, lines = engine.go(30, 500)
                    self.assertIn(move, legal)
                    self.assertEqual(sum(line.startswith("bestmove ") for line in lines), 1)
                    engine.position([])
                move, _, _, lines = engine.go(300, 2000)
                self.assertIn(move, legal)
                self.assertNotIn("info string deadline-fallback", lines)

    def test_summary_handles_incomplete_pairs(self):
        games = [dict(score=1, candidate_side="Attacker", reason="corner-escape")]
        result = summarize(games, 42)
        self.assertIsNone(result["by_side"]["Defender"]["score_rate"])
        self.assertIsNone(result["paired_bootstrap_95"])
        self.assertEqual(paired_interval([1, 0] * 25), [0.5, 0.5])
        self.assertIsNone(paired_interval([]))
        self.assertIsNone(summarize([], 42)["score_rate"])

    def test_teardown_terminates_a_stuck_engine_and_child(self):
        import sys
        import time

        script = """import subprocess,sys,time
subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
for line in sys.stdin:
    command=line.strip()
    if command=='uci': print('uciok',flush=True)
    elif command=='isready': print('readyok',flush=True)
    elif command=='quit': time.sleep(30)
"""
        engine = Engine([sys.executable, "-u", "-c", script])
        started = time.perf_counter()
        engine.close()
        self.assertIsNotNone(engine.proc.poll())
        self.assertFalse(engine.reader.is_alive())
        self.assertLess(time.perf_counter() - started, 8)

    def test_training_minimum_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "data.jsonl"
            output = Path(folder) / "model.json"
            data.write_text(
                "\n".join(json.dumps(dict(game=str(i), target=1)) for i in range(2))
            )
            r = subprocess.run(
                [
                    __import__("sys").executable,
                    str(ROOT / "python/train.py"),
                    "--data",
                    str(data),
                    "--kind",
                    "nnue",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("1,000", r.stdout + r.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
