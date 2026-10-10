"""Bundle runnable Windows binaries, trained weights, source/docs and evidence."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from transport import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "dist/skallagrim-windows.zip")
    p.add_argument("--with-data", action="store_true")
    p.add_argument("--with-matches", action="store_true")
    p.add_argument("--matches-root", type=Path, default=ROOT / "runs/final-gauntlets")
    args = p.parse_args()
    files = {}
    for name in ["skallagrim.exe", "skallagrim-gui.exe", "skallagrim-referee.exe"]:
        files[ROOT / "target/release" / name] = name
    for name in [
        "README.md",
        "CHANGELOG.md",
        "PROJECT_NOTES.md",
        "LICENSE",
        "Cargo.toml",
        "Cargo.lock",
    ]:
        files[ROOT / name] = name
    for folder in ["models", "docs", "src", "tests", "python", "benchmarks", ".github"]:
        for path in (ROOT / folder).rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                files[path] = str(path.relative_to(ROOT)).replace("\\", "/")
    if args.with_data:
        for name in ["training-audited.jsonl", "training-final-audit.audit.json"]:
            files[ROOT / "runs" / name] = "evidence/" + name
    if args.with_matches:
        if not args.matches_root.is_dir():
            raise FileNotFoundError(args.matches_root)
        for path in args.matches_root.rglob("*"):
            if path.is_file():
                files[path] = "evidence/final-gauntlets/" + str(
                    path.relative_to(args.matches_root)
                ).replace("\\", "/")
        checkpoints = set()
        for manifest in args.matches_root.rglob("manifest.json"):
            for entry in json.loads(manifest.read_text(encoding="utf8"))[
                "engines"
            ].values():
                for artifact in entry["sha256"]:
                    path = Path(artifact)
                    if artifact.startswith("external/") and path.suffix in {
                        ".pt",
                        ".ckpt",
                    }:
                        checkpoints.add(path)
        for checkpoint in checkpoints:
            for path in [checkpoint, checkpoint.with_suffix(".meta.json")]:
                files[ROOT / path] = "evidence/opponent-artifacts/" + path.as_posix()
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".tmp")
    with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        checksums = {}
        for path, name in files.items():
            out.write(path, name)
            checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.parent == ROOT / "target/release":
                duplicate = "target/release/" + path.name
                out.write(path, duplicate)
                checksums[duplicate] = checksums[name]
        out.writestr(
            "checksums.json",
            json.dumps(checksums),
        )
    staging.replace(args.output)
    with args.output.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        digest + "  " + args.output.name + "\n", encoding="utf8"
    )
    print(args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
