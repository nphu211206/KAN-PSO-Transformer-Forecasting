"""Copy the audited allowlist into a private release-staging directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def stage(manifest_path: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty staging directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        source = ROOT / relative
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(manifest_path, output / manifest_path.name)
    (output / "STAGING_ONLY.txt").write_text(
        "This package is private release staging. It is not licensed, deposited, "
        "or assigned a DOI. Verify author data, licenses and redistribution policy "
        "before publication.\n",
        encoding="utf-8",
    )
    print(f"Staged {len(manifest['files'])} allowlisted files in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "PUBLIC_RELEASE_MANIFEST.json")
    parser.add_argument("--output", type=Path, default=ROOT / "release_staging_v13")
    args = parser.parse_args()
    stage(args.manifest, args.output)
