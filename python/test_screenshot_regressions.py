"""Regressions for referee recovery, seeded replay and audited labels."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import audit_dataset
from pipeline import replay_sources
from referee import MatchTimeout, Referee


class ScreenshotRegressions(unittest.TestCase):
    def test_match_timeout_keeps_synchronized_connection_usable(self):
        ref = Referee.__new__(Referee)
        ref.failed = False
        ref.sequence = 0
        ref.state = {}
        ref.latencies = []
        ref.proc = Mock()
        outcome = {"winner": "Defender", "reason": "timeout"}
        ref.replies = Mock()
        ref.replies.get.side_effect = [
            json.dumps(
                {
                    "id": 1,
                    "ok": False,
                    "error": "clock expired",
                    "state": {"outcome": outcome},
                }
            ),
            json.dumps(
                {
                    "id": 2,
                    "ok": True,
                    "state": {"outcome": outcome},
                    "result": {"outcome": outcome},
                }
            ),
            json.dumps(
                {
                    "id": 3,
                    "ok": True,
                    "state": {"outcome": None},
                    "result": {"outcome": None},
                }
            ),
        ]
        with self.assertRaises(MatchTimeout):
            ref.call("play", move="a1a2")
        self.assertFalse(ref.failed)
        self.assertEqual(ref.snapshot()["outcome"], outcome)
        self.assertIsNone(ref.position([])["outcome"])

    def test_transport_timeout_still_poisons_connection(self):
        ref = Referee.__new__(Referee)
        ref.failed = False
        ref.latencies = []
        ref.sequence = 0
        ref.state = {}
        ref.proc = Mock()
        ref.proc.pid = 1
        ref._call = Mock(side_effect=TimeoutError("transport stalled"))
        with tempfile.TemporaryDirectory() as tmp:
            ref.log_dir = Path(tmp)
            with self.assertRaises(TimeoutError):
                ref.call("snapshot")
            self.assertTrue(ref.failed)
            with self.assertRaisesRegex(RuntimeError, "unusable"):
                ref.call("snapshot")

    def test_second_seeded_generation_excludes_previous_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in range(3):
                folder = root / f"generation-{n:04d}"
                folder.mkdir()
                (folder / "selfplay.jsonl").touch()
            current = root / "generation-0001"
            self.assertEqual(
                replay_sources(root, current, True), [current / "selfplay.jsonl"]
            )
            self.assertEqual(len(replay_sources(root, current, False)), 2)

    def test_seeded_second_generation_retains_exact_1250_game_split(self):
        from dataset import DatasetIndex, make_replay_sqlite
        from campaign_state import role_split

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for generation in range(2):
                folder = root / f"generation-{generation:04d}"
                folder.mkdir()
                rows = [
                    dict(
                        game=f"{generation}:{i}",
                        fen="fixture a",
                        target=1,
                        league={
                            "candidate_side": "Attacker" if i % 2 == 0 else "Defender"
                        },
                    )
                    for i in range(1250)
                ]
                (folder / "selfplay.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
            current = root / "generation-0001"
            replay = current / "replay.sqlite"
            selection = make_replay_sqlite(
                replay_sources(root, current, True), replay, 1250, 10_000_000, 1250
            )
            self.assertEqual(selection["games"], 1250)
            index = DatasetIndex(replay)
            split = role_split(index, current / "split.json", 1000, 250, 42)
            self.assertEqual(len(split["training"]), 1000)
            self.assertEqual(len(split["validation"]), 250)
            self.assertTrue(all(game.startswith("1:") for game in index.groups))

    def test_audit_checks_every_target_against_verified_winner(self):
        outcome = {"winner": "Defender", "reason": "king-captured"}
        for targets, expected in [([-1, -1], 2), ([-1, 1], 0), ([1, -1], 0)]:
            with self.subTest(targets=targets), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "data.jsonl"
                output = root / "audited.jsonl"
                rows = [
                    {
                        "game": "g",
                        "fen": f"fen{i}",
                        "legal": ["a1a2"],
                        "played_move": "a1a2",
                        "target": target,
                        "outcome": outcome,
                    }
                    for i, target in enumerate(targets)
                ]
                source.write_text("".join(json.dumps(row) + "\n" for row in rows))
                engine = Mock()
                engine.__enter__ = Mock(return_value=engine)
                engine.__exit__ = Mock(return_value=False)
                engine.snapshot.return_value = {"fen": "fen0", "moves": ["a1a2"]}
                engine.play.side_effect = [
                    {"fen": "fen1", "moves": ["a1a2"], "outcome": None},
                    {"fen": "terminal", "moves": [], "outcome": outcome},
                ]
                with (
                    patch.object(audit_dataset, "Engine", return_value=engine),
                    patch(
                        "sys.argv",
                        [
                            "audit_dataset",
                            "--data",
                            str(source),
                            "--output",
                            str(output),
                        ],
                    ),
                ):
                    audit_dataset.main()
                self.assertEqual(len(output.read_text().splitlines()), expected)
                if not expected:
                    self.assertIn(
                        "target",
                        json.loads(output.with_suffix(".audit.json").read_text())[
                            "rejected"
                        ][0]["reason"],
                    )


if __name__ == "__main__":
    unittest.main()
