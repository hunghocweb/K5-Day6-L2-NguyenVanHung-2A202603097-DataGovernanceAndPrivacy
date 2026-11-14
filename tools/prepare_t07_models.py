"""Prepare T07 weights online, or verify local files with --offline (no downloads)."""
import argparse
import json
from pathlib import Path
import urllib.request

from src.common.io import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "reports/t07_models.json").read_text(encoding="utf-8"))
    for kind in ("face", "plate"):
        spec = inventory[kind]
        target = (root / spec["path"]).resolve()
        if not target.is_relative_to((root / "models_optional").resolve()):
            raise ValueError("Model path must stay within models_optional")
        if target.exists():
            if sha256_file(target) != spec["sha256"]:
                raise ValueError(f"Existing model checksum mismatch: {target}")
            print(f"{kind}: local SHA256 verified")
            continue
        if args.offline:
            raise FileNotFoundError(f"Missing {kind} model: {target}; offline verification never downloads")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(spec["source"], timeout=60) as response, partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if sha256_file(partial) != spec["sha256"]:
            raise ValueError(f"Downloaded model checksum mismatch; not installed: {partial}")
        partial.replace(target)
        print(f"{kind}: downloaded and SHA256 verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
