import unittest
from pathlib import Path
from unittest.mock import patch

import pin_campaign_baseline as baseline


class BaselineTests(unittest.TestCase):
    def build(self, promoted=False, contradictory=False, running=False):
        run = Path("fixture-campaign").resolve()

        def read(path):
            relative = path.relative_to(run).as_posix()
            if relative == "status.json":
                return {
                    "stage": "nnue" if running else "complete",
                    "state": "running" if running else "complete",
                    "promoted": {"nnue": promoted, "cnn": promoted},
                }
            if relative == "completed-stages.json":
                return ["nnue", "cnn", "external-gauntlets", "replay-audit"]
            if relative == "replay-audit.json":
                return {"replay_verified": True, "games": 750}
            if relative == "campaign.json":
                return {"files": {}}
            if relative == "runtime/benchmarks/engines.json":
                return {
                    "engines": {bot: {"options": {}} for bot in ["iota", "beta", "rho"]}
                }
            if relative.endswith("complete.json"):
                bots = ["rho"] if relative.startswith("cnn/") else ["iota", "beta"]
                return {
                    "accepted": not promoted if contradictory else promoted,
                    "gate": {
                        f"{bot}/{opponent}": {}
                        for bot in bots
                        for opponent in ["champion", "held-out"]
                    },
                }
            if relative.endswith("meta.json"):
                return {"model_artifacts": {}}
            raise AssertionError(relative)

        with (
            patch.object(baseline, "read", side_effect=read),
            patch.object(
                baseline,
                "pin",
                side_effect=lambda p: {"path": str(p), "sha256": "fixture"},
            ),
            patch.object(baseline, "gate_accepted", return_value=promoted),
        ):
            return baseline.build_baseline(run)

    def test_running_campaign_refused(self):
        with self.assertRaisesRegex(ValueError, "must finish"):
            self.build(running=True)

    def test_rejected_candidate_uses_frozen_incumbent(self):
        result = self.build()
        for bot, kind in [("iota", "nnue"), ("beta", "nnue"), ("rho", "cnn")]:
            option = "CNNFile" if kind == "cnn" else "NNUEFile"
            self.assertIn(
                f"runtime/models/{kind}.json",
                result["config"]["engines"][bot]["options"][option].replace("\\", "/"),
            )

    def test_accepted_candidate_is_selected(self):
        result = self.build(promoted=True)
        self.assertIn(
            "generation-0000",
            result["config"]["engines"]["iota"]["options"]["NNUEFile"],
        )

    def test_contradictory_promotion_refused(self):
        with self.assertRaisesRegex(ValueError, "Contradictory"):
            self.build(contradictory=True)


if __name__ == "__main__":
    unittest.main()
