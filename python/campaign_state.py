"""Durable compressed game transactions and pinned, role-stratified splits."""

import gzip
import json
import os
import random
import time
from pathlib import Path
from records import save
from league import digest


def atomic_gzip(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(
                json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
            )
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)


class Collection:
    def __init__(self, directory, configuration, validate=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = configuration
        path = self.directory / "config.json"
        if path.exists() and json.loads(path.read_text()) != configuration:
            raise ValueError("Collection resume rejected: changed configuration")
        save(path, configuration)
        self.records = []
        self.completed = 0
        self.roles = {"Attacker": 0, "Defender": 0}
        self.elapsed = 0.0
        self.fallbacks = 0
        self.exclusions = {}
        self.outcomes = {}
        for path in sorted(self.directory.glob("attempt-*.json.gz")):
            with gzip.open(path, "rt", encoding="utf8") as stream:
                record = json.load(stream)
            if record["configuration"] != digest(configuration):
                raise ValueError("Collection transaction configuration changed")
            expected = record.pop("sha256")
            if digest(record) != expected:
                raise ValueError("Collection transaction altered")
            record["sha256"] = expected
            if validate:
                validate(record["audit"])
            self._accept(record)

    def _accept(self, record):
        if self.records and record["attempt"] <= self.records[-1]["attempt"]:
            raise ValueError("Duplicate or reordered collection transaction")
        self.records.append(
            {k: record[k] for k in ("attempt", "accepted", "role", "sha256")}
        )
        self.elapsed += record["seconds"]
        self.fallbacks += record["fallbacks"]
        if record["accepted"]:
            self.completed += 1
            self.roles[record["role"]] += 1
            outcome = record["audit"].get("outcome") or {}
            key = outcome.get("winner") or "Draw"
            self.outcomes[key] = self.outcomes.get(key, 0) + 1
        else:
            reason = record["exclusion"]
            self.exclusions[reason] = self.exclusions.get(reason, 0) + 1

    @property
    def next_attempt(self):
        return self.records[-1]["attempt"] + 1 if self.records else 0

    def commit(self, attempt, role, samples, audit, seconds, fallbacks, exclusion):
        accepted = bool(samples) and samples[0].get("target") is not None
        record = dict(
            configuration=digest(self.config),
            attempt=attempt,
            role=role,
            accepted=accepted,
            samples=samples if accepted else [],
            audit=audit,
            seconds=seconds,
            fallbacks=fallbacks,
            exclusion=exclusion,
        )
        record["sha256"] = digest(record)
        path = self.directory / f"attempt-{attempt:06}.json.gz"
        if path.exists():
            raise ValueError("Attempt already committed")
        atomic_gzip(path, record)
        self._accept(record)
        self.progress()

    def progress(self):
        result = dict(
            completed=self.completed,
            requested=self.config["games"],
            roles=self.roles,
            attempts=len(self.records),
            next_attempt=self.next_attempt,
            exclusions=self.exclusions,
            outcomes=self.outcomes,
            seconds=self.elapsed,
            deadline_fallbacks=self.fallbacks,
            games_per_hour=self.completed * 3600 / self.elapsed
            if self.elapsed
            else None,
            updated=time.time(),
        )
        if self.completed >= 50:
            result["remaining_hours_estimate"] = (
                (self.config["games"] - self.completed)
                * self.elapsed
                / self.completed
                / 3600
            )
        save(self.directory.parent / "collection-progress.json", result)
        return result

    def assemble(self, samples_path, audits_path, budget=None):
        samples_path, audits_path = Path(samples_path), Path(audits_path)
        # Compressed sources remain authoritative; rebuilding cannot append duplicates.
        sample_tmp = samples_path.with_suffix(samples_path.suffix + ".tmp")
        audit_tmp = audits_path.with_suffix(audits_path.suffix + ".tmp")
        with (
            gzip.open(sample_tmp, "wt", encoding="utf8") as samples,
            gzip.open(audit_tmp, "wt", encoding="utf8") as audits,
        ):
            for entry in self.records:
                if budget:
                    budget.check()
                path = self.directory / f"attempt-{entry['attempt']:06}.json.gz"
                with gzip.open(path, "rt", encoding="utf8") as stream:
                    record = json.load(stream)
                expected = record.pop("sha256")
                if digest(record) != expected:
                    raise ValueError("Collection transaction altered during assembly")
                for row in record["samples"]:
                    samples.write(json.dumps(row, separators=(",", ":")) + "\n")
                audits.write(json.dumps(record["audit"], separators=(",", ":")) + "\n")
        sample_tmp.replace(samples_path)
        audit_tmp.replace(audits_path)


def role_split(index, path, training_games, validation_games, seed):
    if training_games % 2 or validation_games % 2:
        raise ValueError("Role-balanced split needs even partition counts")
    groups = {
        role: sorted(game for game in index.groups if index.roles.get(game) == role)
        for role in ("Attacker", "Defender")
    }
    if sum(map(len, groups.values())) != training_games + validation_games:
        raise ValueError("Split count does not match requested completed games")
    train, valid = [], []
    rng = random.Random(seed)
    for role, games in groups.items():
        if len(games) != (training_games + validation_games) // 2:
            raise ValueError(f"Unbalanced candidate roles: {role}")
        rng.shuffle(games)
        valid.extend(games[: validation_games // 2])
        train.extend(games[validation_games // 2 :])
    value = dict(
        format="skallagrim-split-v1",
        data_sha256=index.sha256,
        seed=seed,
        training=sorted(train),
        validation=sorted(valid),
    )
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Split resume rejected: changed data or split")
    save(path, value)
    return value
