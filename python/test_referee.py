"""Process-level checks for the actual dedicated Rust authority."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from referee import Referee


class RefereeTests(unittest.TestCase):
    def test_dynamic_fort_automatic_ruling_certificate_and_replay(self):
        # Static proof fails because the corner-hostile wall is breachable;
        # every attacker response nevertheless admits the king's escape.
        fen = "1kda7/dd9/a1a8/11/11/11/11/11/11/11/11 a"
        with self.referee() as ref:
            state = ref.position(fen=fen)
            self.assertIsNone(state["outcome"])
            ref.call("analyze", claim="fort", nodes=100000, milliseconds=10000)
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                state = ref.snapshot()
                if (state.get("analysis") or {}).get("claim") == "fort":
                    break
                time.sleep(0.01)
            self.assertEqual(state["analysis"]["status"], "proven")
            self.assertEqual(state["outcome"]["reason"], "exit-fort-dynamic")
            replay = ref.export()
            self.assertEqual(replay["referee"]["version"], 2)
            self.assertEqual(ref.call("validate", replay=replay)["valid"], True)
            cert = replay["referee"]["events"][1]["evidence"]["proof"]["certificate"]
            cert["nodes"][0]["edges"].pop()
            with self.assertRaises(ValueError):
                ref.call("validate", replay=replay)

    def referee(self):
        return Referee(os.environ.get("SKALLAGRIM_REFEREE"))

    def test_owns_moves_history_and_rejects_stale_revision(self):
        with self.referee() as ref:
            original = ref.snapshot()
            state = ref.play(original["moves"][0])
            self.assertNotEqual(state["fen"], original["fen"])
            self.assertGreater(len(state["history"]), len(original["history"]))
            with self.assertRaises(ValueError):
                ref.call("play", move=state["moves"][0], revision=original["revision"])
            self.assertEqual(ref.snapshot()["fen"], state["fen"])

    def test_clock_runs_during_background_proof(self):
        with self.referee() as ref:
            ref.start_clock(30)
            ref.call("analyze", claim="draw", nodes=100000, milliseconds=1000)
            time.sleep(0.12)
            state = ref.snapshot()
            self.assertEqual(state["outcome"]["reason"], "timeout")
            with self.assertRaises(TimeoutError):
                ref.play("d1d2")
            events = ref.export()["referee"]["events"]
            self.assertTrue(
                any(
                    e["kind"] == "ruling" and e["evidence"]["source"] == "clock"
                    for e in events
                )
            )

    def test_unknown_fort_never_ends_game_and_human_requires_note(self):
        with self.referee() as ref:
            ref.call("analyze", claim="fort")
            deadline = time.monotonic() + 2
            while not ref.snapshot()["analysis"] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(ref.state["analysis"]["status"], "unknown")
            self.assertIsNone(ref.state["outcome"])
            with self.assertRaises(ValueError):
                ref.call("adjudicate", claim="draw", operator="Tester", reason="")
            ref.call(
                "adjudicate",
                claim="draw",
                operator="Tester",
                reason="Human judgment after inconclusive certificate search",
            )
            event = ref.export()["referee"]["events"][-1]
            self.assertEqual(event["evidence"]["operator"], "Tester")
            self.assertEqual(event["evidence"]["analysis"]["status"], "unknown")
            self.assertTrue(ref.call("validate", replay=ref.export())["valid"])

    def test_late_analysis_cannot_attach_to_new_position(self):
        with self.referee() as ref:
            ref.call("analyze", claim="draw", nodes=100000, milliseconds=200)
            ref.position([])
            time.sleep(0.3)
            self.assertIsNone(ref.snapshot()["analysis"])
            self.assertIsNone(ref.state["outcome"])

    def test_rulings_are_saved_before_acknowledgment_and_survive_new_game(self):
        with tempfile.TemporaryDirectory() as directory:
            with Referee(log_dir=directory) as ref:
                ref.position([])
                ref.call(
                    "adjudicate",
                    claim="draw",
                    operator="Alice",
                    reason="Human decision after evidence review",
                )
                replays = [
                    json.loads(path.read_text(encoding="utf8"))
                    for path in Path(directory).glob("*.json")
                ]
                ruled = [replay for replay in replays if replay["outcome"]]
                self.assertEqual(len(ruled), 1)
                self.assertTrue(ref.call("validate", replay=ruled[0])["valid"])
                ref.position([])
                replays = [
                    json.loads(path.read_text(encoding="utf8"))
                    for path in Path(directory).glob("*.json")
                ]
                self.assertTrue(any(replay["outcome"] for replay in replays))

    def test_draw_witness_is_checked_on_replay(self):
        with self.referee() as ref:
            ref.position([], fen="5k5/11/11/a10/11/11/11/11/11/11/11 d")
            ref.call("analyze", claim="draw", nodes=1000, milliseconds=500)
            deadline = time.monotonic() + 2
            while not ref.snapshot()["analysis"] and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(ref.state["analysis"]["status"], "finish-reachable")
            replay = ref.export()
            self.assertTrue(ref.call("validate", replay=replay)["valid"])
            analysis = next(
                e for e in replay["referee"]["events"] if e["kind"] == "analysis"
            )
            analysis["evidence"]["proof"]["witness"] = ["a1a11"]
            with self.assertRaises(ValueError):
                ref.call("validate", replay=replay)

    def test_pause_after_clock_expiry_refreshes_state_without_stale_revision_error(
        self,
    ):
        with self.referee() as ref:
            ref.start_clock(20)
            time.sleep(0.06)
            state = ref.call("pause", reason="Human opened review panel")
            self.assertEqual(state["outcome"]["reason"], "timeout")

    def test_late_forfeit_keeps_authoritative_clock_result(self):
        with self.referee() as ref:
            ref.start_clock(20)
            time.sleep(0.06)
            state = ref.call(
                "forfeit", reason="engine-failure", evidence="Player crashed"
            )
            self.assertEqual(state["outcome"]["reason"], "timeout")

    def test_unicode_human_identity_round_trips(self):
        with self.referee() as ref:
            name = "Referee \u03b1 \u2014 \u092a\u0930\u0940\u0915\u094d\u0937\u0923"
            reason = "Human review \u2014 unresolved position"
            ref.call("adjudicate", claim="draw", operator=name, reason=reason)
            ruling = ref.export()["referee"]["events"][-1]["evidence"]
            self.assertEqual(ruling["operator"], name)
            self.assertEqual(ruling["reason"], reason)

    def test_child_is_reaped_on_close(self):
        ref = self.referee()
        process = ref.proc
        ref.close()
        self.assertIsNotNone(process.poll())


if __name__ == "__main__":
    unittest.main()
