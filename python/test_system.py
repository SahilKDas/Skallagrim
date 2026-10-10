"""Integration checks for protocol atomicity, cancellation, neural parity gates."""

import json
import subprocess
import unittest
import tempfile
from pathlib import Path
from transport import Engine, ROOT
from benchmark import summarize, paired_interval


class SystemTests(unittest.TestCase):
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

    def test_summary_handles_incomplete_pairs(self):
        games = [dict(score=1, candidate_side="Attacker", reason="corner-escape")]
        result = summarize(games, 42)
        self.assertIsNone(result["by_side"]["Defender"]["score_rate"])
        self.assertIsNone(result["paired_bootstrap_95"])
        self.assertEqual(paired_interval([1, 0] * 25), [0.5, 0.5])

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
