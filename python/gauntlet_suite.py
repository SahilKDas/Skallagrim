"""Run all fifteen required 50-game/300-ms pairings, optionally on three CPUs."""

import argparse
import subprocess
import sys
from pathlib import Path
from transport import ROOT

PAIRINGS = {
    "rho": ["taflzero", "opentafl", "luizz", "nicholas", "iota", "beta"],
    "iota": ["taflzero", "opentafl", "luizz", "nicholas", "beta"],
    "beta": ["taflzero", "opentafl", "luizz", "nicholas"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpus", nargs=3, type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-unaudited", action="store_true")
    p.add_argument("--load-note", default="Ambient desktop load uncontrolled")
    args = p.parse_args()
    if args.cpus and len(set(args.cpus)) != 3:
        raise SystemExit("Use three distinct logical CPUs")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    running = []
    logs = []
    try:
        for i, (candidate, opponents) in enumerate(PAIRINGS.items()):
            cmd = [
                sys.executable,
                str(ROOT / "python/benchmark.py"),
                "--candidate",
                candidate,
                "--opponents",
                *opponents,
                "--output",
                str(args.output / candidate),
                "--load-note",
                args.load_note,
            ]
            if args.cpus:
                cmd += ["--cpu", str(args.cpus[i])]
            if args.resume:
                cmd += ["--resume"]
            if args.allow_unaudited:
                cmd += ["--allow-unaudited"]
            log = (args.output / (candidate + ".campaign.log")).open(
                "a" if args.resume else "w"
            )
            logs.append(log)
            process = subprocess.Popen(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            running.append(process)
            if not args.cpus and process.wait() != 0:
                raise RuntimeError(f"{candidate} campaign failed; see {log.name}")
        for process in running:
            if process.wait() != 0:
                raise RuntimeError("Campaign failed; see per-candidate logs")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "python/report.py"),
                "--root",
                str(args.output),
                "--output",
                str(ROOT / "benchmarks/RESULTS.md"),
                "--require-complete",
            ],
            cwd=ROOT,
            check=True,
        )
    finally:
        for process in running:
            if process.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    )
                else:
                    process.send_signal(__import__("signal").SIGINT)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
