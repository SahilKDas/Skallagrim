import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from records import save


class AtomicRecordTests(unittest.TestCase):
    def lock_error(self):
        error = PermissionError("Windows transient sharing denial")
        error.winerror = 5
        return error

    def test_transient_windows_lock_preserves_old_file_until_atomic_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.json"
            save(path, {"committed": 1210})
            import records

            replace = records.os.replace
            attempts = []

            def locked_once(source, destination):
                attempts.append(1)
                self.assertEqual(json.loads(path.read_text()), {"committed": 1210})
                if len(attempts) == 1:
                    raise self.lock_error()
                replace(source, destination)

            with (
                patch("records.os.replace", side_effect=locked_once),
                patch("records.time.sleep"),
            ):
                save(path, {"committed": 1211})
            self.assertEqual(json.loads(path.read_text()), {"committed": 1211})
            self.assertEqual(len(attempts), 2)
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_persistent_denial_is_bounded_and_keeps_committed_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.json"
            save(path, {"committed": 1210})
            with (
                patch("records.os.replace", side_effect=self.lock_error()) as replace,
                patch("records.time.sleep"),
            ):
                with self.assertRaises(PermissionError):
                    save(path, {"committed": 1211})
            self.assertEqual(replace.call_count, 9)
            self.assertEqual(json.loads(path.read_text()), {"committed": 1210})
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_unrelated_permission_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.json"
            with (
                patch(
                    "records.os.replace", side_effect=PermissionError("ACL denial")
                ) as replace,
                patch("records.time.sleep") as sleep,
            ):
                with self.assertRaises(PermissionError):
                    save(path, {"committed": 1211})
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
