"""Pair-sharded collection; identical pinned searches and global game identities."""

import argparse
import ctypes
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from campaign_state import Collection, atomic_gzip
from league import digest, pin, verify_manifest
from records import save
from referee import Referee
from resources import StorageBudget
from transport import ROOT


def set_worker_resources(cpu):
    if not 0 <= cpu < (os.cpu_count() or 1):
        raise ValueError("Worker CPU out of range")
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        if not kernel.SetPriorityClass(
            handle, 0x40
        ) or not kernel.SetProcessAffinityMask(handle, 1 << cpu):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.sched_setaffinity(0, {cpu})


def worker_config(league, cpus, index, exploration=0.15):
    per_role, extra = divmod(league["games"] // 2, len(cpus))
    return dict(
        manifest=digest(league),
        games=2 * (per_role + (index < extra)),
        exploration=exploration,
        cpu=cpus[index],
        version="skallagrim-collection-v1",
        worker_index=index,
        worker_count=len(cpus),
    )


def read_record(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        record = json.load(stream)
    expected = record.pop("sha256")
    if digest(record) != expected:
        raise ValueError(f"Altered committed game: {path}")
    record["sha256"] = expected
    return record


def migrate(source, folder, league, cpus, validate, budget=None):
    """Copy immutable audited transactions; only execution configuration changes."""
    source, folder = Path(source), Path(folder)
    source_league = json.loads((source / "league.json").read_text())
    if source_league != league:
        raise ValueError("Migration rejected: changed league or schedule")
    config = json.loads((source / "transactions/config.json").read_text())
    expected = dict(
        manifest=digest(league),
        games=league["games"],
        exploration=0.15,
        cpu=6,
        version="skallagrim-collection-v1",
    )
    if config != expected:
        raise ValueError("Migration rejected: unsupported source configuration")
    provenance = []
    configurations = [digest(worker_config(league, cpus, i)) for i in range(len(cpus))]
    for path in sorted((source / "transactions").glob("attempt-*.json.gz")):
        if budget:
            budget.check()
        record = read_record(path)
        if record["configuration"] != digest(config):
            raise ValueError("Source transaction configuration mismatch")
        attempt = record["attempt"]
        plan = league["schedule"][attempt]
        if (
            record["audit"]["league"] != plan
            or record["role"] != plan["candidate_side"]
        ):
            raise ValueError("Source transaction schedule mismatch")
        if not validate(record["audit"]):
            raise ValueError("Source game failed referee audit")
        worker = plan["pair"] % len(cpus)
        destination = folder / f"worker-{worker}" / "transactions" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        original_hash = record["sha256"]
        record["configuration"] = configurations[worker]
        record["migration"] = dict(
            source=pin(path),
            original_sha256=original_hash,
            original_configuration=digest(config),
        )
        record.pop("sha256")
        record["sha256"] = digest(record)
        if destination.exists():
            if read_record(destination) != record:
                raise ValueError("Migration destination changed")
        else:
            atomic_gzip(destination, record)
        provenance.append(
            dict(
                source=pin(path),
                worker=worker,
                attempt=attempt,
                destination_sha256=record["sha256"],
            )
        )
    result = dict(
        format="skallagrim-collection-migration-v1",
        source=str(source.resolve()),
        source_league=pin(source / "league.json"),
        transactions=provenance,
        completed=sum(
            read_record(Path(x["source"]["path"]))["accepted"] for x in provenance
        ),
    )
    journal = folder / "migration.json"
    if journal.exists() and json.loads(journal.read_text()) != result:
        raise ValueError("Migration inputs changed on resume")
    save(journal, result)
    return result


def assemble(folder, league, cpus, output, validate, budget=None):
    output = Path(output)
    samples_path = output.with_suffix(".jsonl.gz")
    audit_path = output.with_suffix(".referee.jsonl.gz")
    sample_tmp = samples_path.with_suffix(samples_path.suffix + ".tmp")
    audit_tmp = audit_path.with_suffix(audit_path.suffix + ".tmp")
    records = sorted(
        (
            path
            for i in range(len(cpus))
            for path in (folder / f"worker-{i}/transactions").glob("attempt-*.json.gz")
        ),
        key=lambda p: p.name,
    )
    seen_attempts, seen_games = set(), set()
    configurations = [digest(worker_config(league, cpus, i)) for i in range(len(cpus))]
    completed, roles = 0, {"Attacker": 0, "Defender": 0}
    with (
        gzip.open(sample_tmp, "wt", encoding="utf8") as samples,
        gzip.open(audit_tmp, "wt", encoding="utf8") as audits,
    ):
        for path in records:
            if budget:
                budget.check()
            record = read_record(path)
            attempt = record["attempt"]
            plan = league["schedule"][attempt]
            worker = plan["pair"] % len(cpus)
            if (
                path.parent.parent.name != f"worker-{worker}"
                or record["configuration"] != configurations[worker]
            ):
                raise ValueError("Wrong worker or changed transaction configuration")
            if attempt in seen_attempts or record["audit"]["game"] in seen_games:
                raise ValueError("Duplicate global campaign game")
            seen_attempts.add(attempt)
            seen_games.add(record["audit"]["game"])
            if (
                record["role"] != plan["candidate_side"]
                or record["audit"]["league"] != plan
            ):
                raise ValueError("Transaction disagrees with pinned schedule")
            if not validate(record["audit"]):
                raise ValueError("Committed replay failed final referee audit")
            if record["accepted"]:
                if not record["samples"] or any(
                    row["target"] is None or row["game"] != record["audit"]["game"]
                    for row in record["samples"]
                ):
                    raise ValueError("Invalid training labels or game identity")
                completed += 1
                roles[record["role"]] += 1
                for row in record["samples"]:
                    samples.write(json.dumps(row, separators=(",", ":")) + "\n")
            elif record["samples"]:
                raise ValueError("Excluded transaction retained training labels")
            audits.write(json.dumps(record["audit"], separators=(",", ":")) + "\n")
    if completed != league["games"] or set(roles.values()) != {league["games"] // 2}:
        raise ValueError(
            "Parallel collection did not meet exact game/role requirements"
        )
    sample_tmp.replace(samples_path)
    audit_tmp.replace(audit_path)
    save(
        folder / "assembly-audit.json",
        dict(
            games=completed,
            roles=roles,
            attempts=len(records),
            corpus=pin(samples_path),
            replays=pin(audit_path),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--max-storage-gib", type=float, default=16)
    args = parser.parse_args()
    if len(args.cpus) < 2 or len(set(args.cpus)) != len(args.cpus):
        raise ValueError("Use at least two distinct worker CPUs")
    league = json.loads(args.league.read_text())
    verify_manifest(league)
    folder = args.output.parent / "parallel"
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(
        format="skallagrim-parallel-v1",
        league=pin(args.league),
        cpus=args.cpus,
        source=str(args.source.resolve()) if args.source else None,
    )
    config_path = folder / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Parallel resume rejected: changed CPU layout/source/league")
    save(config_path, config)
    budget = StorageBudget(args.storage_root, int(args.max_storage_gib * 1024**3))
    budget.check()
    with Referee(
        executable=league["referee"]["path"], log_dir=folder / "audit-referee"
    ) as referee:
        validate = lambda audit: referee.call("validate", replay=audit)["valid"]
        migration = (
            migrate(args.source, folder, league, args.cpus, validate, budget)
            if args.source
            else {"completed": 0}
        )
        prior = (
            json.loads((args.output.parent / "collection-progress.json").read_text())
            if (args.output.parent / "collection-progress.json").exists()
            else {}
        )
        previous_parallel_seconds = prior.get("parallel_seconds", 0)
        started = time.monotonic()
        processes, logs = [], []
        try:
            for i, cpu in enumerate(args.cpus):
                worker = folder / f"worker-{i}"
                worker.mkdir(exist_ok=True)
                # Parent-side integrity reconstruction publishes imported progress before launch.
                state = Collection(
                    worker / "transactions", worker_config(league, args.cpus, i)
                )
                state.progress()
                if state.completed > state.config["games"] or any(
                    n > state.config["games"] // 2 for n in state.roles.values()
                ):
                    raise ValueError("Worker imported more than its role quota")
                log = (worker / "collection.log").open("a", encoding="utf8")
                logs.append(log)
                command = [
                    sys.executable,
                    str(ROOT / "python/selfplay.py"),
                    "--games",
                    str(league["games"]),
                    "--movetime",
                    str(league["movetime"]),
                    "--seed",
                    str(league["seed"]),
                    "--max-plies",
                    str(league["max_plies"]),
                    "--max-attempts",
                    str(league["max_attempts"]),
                    "--bot",
                    league["current"]["bot"],
                    "--league",
                    str(args.league),
                    "--output",
                    str(worker / "selfplay.jsonl"),
                    "--resume-collection",
                    "--cpu",
                    str(cpu),
                    "--worker-index",
                    str(i),
                    "--worker-count",
                    str(len(args.cpus)),
                    "--storage-root",
                    str(args.storage_root),
                    "--max-storage-gib",
                    str(args.max_storage_gib),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                )
                processes.append(process)
            save(
                folder / "workers.json",
                [dict(pid=p.pid, cpu=cpu) for p, cpu in zip(processes, args.cpus)],
            )
            while True:
                budget.check()
                rows = [
                    json.loads(
                        (folder / f"worker-{i}/collection-progress.json").read_text()
                    )
                    for i in range(len(args.cpus))
                ]
                completed = sum(row["completed"] for row in rows)
                elapsed = previous_parallel_seconds + time.monotonic() - started
                new_games = completed - migration["completed"]
                progress = dict(
                    completed=completed,
                    requested=league["games"],
                    roles={
                        side: sum(row["roles"][side] for row in rows)
                        for side in ("Attacker", "Defender")
                    },
                    attempts=sum(row["attempts"] for row in rows),
                    exclusions={
                        reason: sum(row["exclusions"].get(reason, 0) for row in rows)
                        for reason in set().union(*(row["exclusions"] for row in rows))
                    },
                    outcomes={
                        reason: sum(
                            row.get("outcomes", {}).get(reason, 0) for row in rows
                        )
                        for reason in set().union(
                            *(row.get("outcomes", {}) for row in rows)
                        )
                    },
                    seconds=sum(row["seconds"] for row in rows),
                    parallel_seconds=elapsed,
                    imported_games=migration["completed"],
                    parallel_completed=new_games,
                    games_per_hour=new_games * 3600 / elapsed if new_games else None,
                    deadline_fallbacks=sum(row["deadline_fallbacks"] for row in rows),
                    workers=rows,
                    cpus=args.cpus,
                    storage_available_bytes=budget.available_bytes(),
                    free_disk_bytes=shutil.disk_usage(args.storage_root).free,
                    updated=time.time(),
                )
                if new_games >= 50:
                    progress["remaining_hours_estimate"] = (
                        (league["games"] - completed) * elapsed / new_games / 3600
                    )
                save(args.output.parent / "collection-progress.json", progress)
                codes = [process.poll() for process in processes]
                if any(code is not None and code != 0 for code in codes):
                    raise RuntimeError(
                        f"Parallel worker failed: {codes}; committed games retained"
                    )
                if all(code is not None for code in codes):
                    break
                time.sleep(1)
            assemble(folder, league, args.cpus, args.output, validate, budget)
        finally:
            for process in processes:
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            capture_output=True,
                            timeout=10,
                        )
                    else:
                        process.terminate()
                    process.wait(timeout=10)
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
