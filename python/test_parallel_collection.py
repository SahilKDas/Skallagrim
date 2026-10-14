import gzip
import json
import tempfile
import unittest
from pathlib import Path

from campaign_state import Collection, atomic_gzip
from league import digest
from parallel_collection import assemble, migrate, worker_config
from records import save


class ParallelCollectionTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        league = dict(
            games=10,
            schedule=[
                dict(
                    pair=i // 2, candidate_side="Attacker" if i % 2 == 0 else "Defender"
                )
                for i in range(100)
            ],
        )
        save(source / "league.json", league)
        config = dict(
            manifest=digest(league),
            games=10,
            exploration=0.15,
            cpu=6,
            version="skallagrim-collection-v1",
        )
        collection = Collection(source / "transactions", config)
        for i in range(5):
            samples, audit = self.game(i, league)
            collection.commit(
                i, league["schedule"][i]["candidate_side"], samples, audit, 1, 0, None
            )
        return source, league

    def game(self, i, league):
        game = f"10000:{i}"
        return [
            dict(
                game=game,
                target=-1.0,
                fen="preserved-position",
                history=["a1a2"],
                policy=[["b1b2", 1]],
            )
        ], dict(
            game=game,
            league=league["schedule"][i],
            outcome=dict(winner="Defender", reason="corner-escape"),
        )

    def test_pair_shards_exact_uneven_quotas_and_preserved_migration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, league = self.fixture(root)
            cpus, folder = [4, 6, 8, 10], root / "parallel"
            self.assertEqual(
                [worker_config(league, cpus, i)["games"] for i in range(4)],
                [4, 2, 2, 2],
            )
            audits = []
            result = migrate(
                source, folder, league, cpus, lambda a: audits.append(a) or True
            )
            self.assertEqual(result["completed"], 5)
            self.assertEqual(len(audits), 5)
            self.assertEqual(
                migrate(source, folder, league, cpus, lambda a: True), result
            )
            collections = [
                Collection(
                    folder / f"worker-{i}/transactions", worker_config(league, cpus, i)
                )
                for i in range(4)
            ]
            for i in range(5, 10):
                samples, audit = self.game(i, league)
                worker = (i // 2) % 4
                collections[worker].commit(
                    i,
                    league["schedule"][i]["candidate_side"],
                    samples,
                    audit,
                    1,
                    0,
                    None,
                )
            output = root / "selfplay.jsonl"
            assemble(folder, league, cpus, output, lambda audit: True)
            with gzip.open(output.with_suffix(".jsonl.gz"), "rt") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(
                [r["game"] for r in rows], [f"10000:{i}" for i in range(10)]
            )
            self.assertEqual(rows[0], self.game(0, league)[0][0])
            assemble(folder, league, cpus, output, lambda audit: True)
            audit = json.loads((folder / "assembly-audit.json").read_text())
            self.assertEqual(audit["roles"], {"Attacker": 5, "Defender": 5})
            self.assertEqual(sum(c.completed for c in collections), 10)

    def test_migration_rejects_schedule_changes_and_bad_audits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, league = self.fixture(root)
            with self.assertRaises(ValueError):
                migrate(source, root / "bad", league, [4, 6], lambda a: False)
            altered = json.loads(json.dumps(league))
            altered["schedule"][0]["pair"] = 12
            with self.assertRaises(ValueError):
                migrate(source, root / "bad", altered, [4, 6], lambda a: True)
            folder = root / "good"
            migrate(source, folder, league, [4, 6], lambda a: True)
            with self.assertRaises(ValueError):
                migrate(source, folder, league, [4, 8], lambda a: True)

    def test_incomplete_duplicate_and_interrupted_commits_cannot_enter_corpus(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, league = self.fixture(root)
            cpus, folder = [4, 6], root / "parallel"
            migrate(source, folder, league, cpus, lambda a: True)
            worker = folder / "worker-0/transactions"
            (worker / "attempt-999999.json.gz.tmp").write_bytes(b"interrupted write")
            self.assertEqual(
                Collection(worker, worker_config(league, cpus, 0)).completed, 3
            )
            with self.assertRaises(ValueError):
                assemble(folder, league, cpus, root / "selfplay.jsonl", lambda a: True)
            original = next(worker.glob("attempt-*.json.gz"))
            with gzip.open(original, "rt") as stream:
                record = json.load(stream)
            atomic_gzip(worker / "attempt-999998.json.gz", record)
            with self.assertRaises(ValueError):
                assemble(folder, league, cpus, root / "selfplay.jsonl", lambda a: True)


if __name__ == "__main__":
    unittest.main()
