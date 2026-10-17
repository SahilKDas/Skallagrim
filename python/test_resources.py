import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from resources import StorageBudget


class StorageBudgetTests(unittest.TestCase):
    def test_exact_nested_hidden_files_and_fresh_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.hidden').mkdir()
            file = root / '.hidden' / 'data'
            file.write_bytes(b'x' * 31)
            budget = StorageBudget(root, 100, reserve_bytes=20)
            with patch('resources.shutil.disk_usage', return_value=shutil._ntuple_diskusage(1000, 100, 900)):
                self.assertEqual(budget.available_bytes(), 69)
                file.write_bytes(b'x' * 99)
                budget.check()
                (root / 'new').write_bytes(b'xx')
                with self.assertRaises(RuntimeError):
                    budget.check()
                file.unlink()
                self.assertEqual(budget.available_bytes(), 98)

    def test_disk_reserve_and_exact_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = StorageBudget(directory, 100, reserve_bytes=20)
            for free, expected in [(19, -1), (20, 0), (21, 1)]:
                with patch('resources.shutil.disk_usage', return_value=shutil._ntuple_diskusage(1000, 1000-free, free)):
                    self.assertEqual(budget.available_bytes(), expected)
                    if free < 20:
                        with self.assertRaises(RuntimeError):
                            budget.check()
                    else:
                        budget.check()

    def test_matches_original_file_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(8):
                folder = root / str(i) / 'nested'
                folder.mkdir(parents=True)
                for j in range(7):
                    (folder / str(j)).write_bytes(b'x' * (i+j))
            expected = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            with patch('resources.shutil.disk_usage', return_value=shutil._ntuple_diskusage(100000, 0, 100000)):
                self.assertEqual(StorageBudget(root, 10000, 0).available_bytes(), 10000-expected)


if __name__ == '__main__':
    unittest.main()
