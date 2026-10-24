"""Blend protocol and corpus-extension partition regression checks."""
import unittest
from unittest.mock import Mock

from alternate_campaign import extension_split
from transport import Engine, ROOT


class HandcraftedTests(unittest.TestCase):
    def test_extension_preserves_old_validation_and_balances_new_roles(self):
        extension = Mock(groups={f"new:{i}": [i] for i in range(1000)},
                         roles={f"new:{i}": "Attacker" if i % 2 == 0 else "Defender" for i in range(1000)})
        old = dict(training=["old-train"], validation=["old-valid"])
        index = Mock(groups={**extension.groups, "old-train": [1001], "old-valid": [1002]}, sha256="hash")
        split = extension_split(old, index, extension, 7)
        self.assertIn("old-valid", split["validation"])
        self.assertNotIn("old-valid", split["training"])
        self.assertIn("old-train", split["training"])
        self.assertEqual(len(split["training"]), 801)
        self.assertEqual(len(split["validation"]), 201)
        self.assertFalse(set(split["training"]) & set(split["validation"]))
        for role in ("Attacker", "Defender"):
            self.assertEqual(sum(extension.roles.get(g) == role for g in split["validation"]), 100)
        extension.groups["old-train"] = [0]
        with self.assertRaisesRegex(ValueError, "reuses"):
            extension_split(old, index, extension, 7)

    def test_all_bots_blend_values_and_reject_invalid_weights_atomically(self):
        # Explicitly use the working build, not a frozen historical engine.
        with Engine(command=[str(ROOT / "target/release/skallagrim.exe")], options={
            "NNUEFile": str(ROOT / "models/nnue.json"),
            "CNNFile": str(ROOT / "models/cnn.json"),
        }) as engine:
            for bot, raw, blended in [("Iota", "nnue", "blended_nnue"),
                                      ("Beta", "nnue", "blended_nnue"),
                                      ("Rho", "cnn", "blended_cnn")]:
                engine.send(f"setoption name Bot value {bot}")
                engine.ready()
                for weight in (0, 25, 100):
                    engine.send(f"setoption name HandcraftedWeight value {weight}")
                    engine.ready()
                    result = engine.evaluate()
                    value = result[raw][0] if raw == "cnn" else result[raw]
                    expected = (1 - weight / 100) * value + weight / 100 * result["handcrafted"]
                    self.assertAlmostEqual(result[blended], expected, places=6)
                    move, _, _, _ = engine.go(30)
                    self.assertIn(move, engine.snapshot()["moves"])
            for value in ("-2", "101", "nan", "25.5"):
                engine.send(f"setoption name HandcraftedWeight value {value}")
                with self.assertRaises(RuntimeError):
                    engine.until("readyok", 1)
                engine.ready()
                self.assertEqual(engine.evaluate()["handcrafted_weight"], 100)

    def test_bot_defaults_follow_switches_and_explicit_override_survives(self):
        with Engine(command=[str(ROOT / "target/release/skallagrim.exe")]) as engine:
            for bot, expected in [("Iota", 25), ("Beta", 0), ("Rho", 0), ("Iota", 25)]:
                engine.send(f"setoption name Bot value {bot}")
                engine.ready()
                self.assertEqual(engine.evaluate()["handcrafted_weight"], expected)
            engine.send("setoption name HandcraftedWeight value 0")
            engine.ready()
            self.assertEqual(engine.evaluate()["handcrafted_weight"], 0)
            engine.send("setoption name HandcraftedWeight value 40")
            engine.send("setoption name Bot value Beta")
            engine.ready()
            self.assertEqual(engine.evaluate()["handcrafted_weight"], 40)
            engine.send("setoption name HandcraftedWeight value -1")
            engine.ready()
            self.assertEqual(engine.evaluate()["handcrafted_weight"], 0)
            engine.send("setoption name Bot value Iota")
            engine.ready()
            self.assertEqual(engine.evaluate()["handcrafted_weight"], 25)


if __name__ == "__main__":
    unittest.main()
