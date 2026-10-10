"""Launch the compiled Java bridge with upstream OpenTafl classes."""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classpath", required=True)
    p.add_argument("--rules", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = args.classpath.split(";" if ";" in args.classpath else os.pathsep)
    cp = os.pathsep.join([str(root / "external/opentafl/bridge"), *paths])
    process = subprocess.Popen(
        [
            "java",
            "-Xms256m",
            "-Xmx256m",
            "-XX:+UseZGC",
            "-XX:ConcGCThreads=1",
            "-XX:ParallelGCThreads=1",
            "-cp",
            cp,
            "OpenTaflBridge",
        ]
    )
    if "SKALLAGRIM_CPU" in os.environ:
        import sys

        sys.path.insert(0, str(root / "python"))
        from transport import pin_process

        pin_process(process, int(os.environ["SKALLAGRIM_CPU"]))
    raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
