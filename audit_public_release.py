"""Verify that a staged public-release tree exactly matches its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> list[str]:
    manifest_path = root / "PUBLIC_RELEASE_MANIFEST.json"
    if not manifest_path.is_file():
        return ["missing PUBLIC_RELEASE_MANIFEST.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in manifest["files"]}
    failures = []
    for relative, entry in expected.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if path.stat().st_size != entry["bytes"]:
            failures.append(f"size mismatch: {relative}")
        if sha256(path) != entry["sha256"]:
            failures.append(f"SHA-256 mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    permitted = set(expected) | {"PUBLIC_RELEASE_MANIFEST.json", "STAGING_ONLY.txt"}
    extras = sorted(actual - permitted)
    if extras:
        failures.append("unexpected files: " + ", ".join(extras))
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("staging_dir", type=Path)
    args = parser.parse_args()
    failures = audit(args.staging_dir)
    if failures:
        raise SystemExit("Public release audit failed:\n- " + "\n- ".join(failures))
    print("Public release audit passed: manifest, hashes, sizes and exclusion boundary are intact")
