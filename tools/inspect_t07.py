"""Inspect T07 inputs without labels; generate a development-only contact sheet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.common.contact_sheet import render_contact_sheet
from src.common.manifest import verify_manifest
from src.common.review import make_review_queue


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("fixtures/t07_redaction"))
    ap.add_argument("--output", type=Path, default=Path("out/t07_inspection"))
    args = ap.parse_args(argv)
    problems = verify_manifest(args.input / "manifest.csv")
    if problems:
        raise ValueError(f"Manifest failed: {problems[:3]}")
    images = pd.read_csv(args.input / "images.csv")
    required = {"image_id", "path", "width", "height", "split"}
    if not required.issubset(images.columns) or images.image_id.duplicated().any():
        raise ValueError("Invalid images.csv schema or duplicate image_id")
    if not set(images.split).issubset({"dev", "eval"}):
        raise ValueError("Unexpected split")
    # Only dev pixels are opened. Eval metadata is counted, never used to tune.
    dev = images.loc[images.split == "dev"].sort_values("image_id").copy()
    for row in dev.itertuples(index=False):
        with Image.open(args.input / row.path) as image:
            if image.size != (row.width, row.height):
                raise ValueError(f"Incorrect image dimensions: {row.image_id}")
            image.verify()
    profile = {
        "track": "t07", "n_images": len(images),
        "split_counts": {str(k): int(v) for k, v in images.split.value_counts().items()},
        "dimensions": images[["width", "height"]].drop_duplicates().to_dict("records"),
        "dev_image_ids": dev.image_id.tolist(), "pixels_inspected": "dev only",
        "manifest_problems": problems, "ground_truth_read": False,
        "human_review_completed": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    dev["score"] = 0.0
    dev["reason"] = "Development input inspection; not a model result or completed human review"
    queue = make_review_queue(dev, "image_id", "score", len(dev), reason_col="reason",
                              slice_col="split", evidence_col="path")
    render_contact_sheet(queue, args.output / "dev_contact_sheet.html", image_root=args.input,
                         thumb_width=640, title="T07 development inputs - no model predictions")
    print(json.dumps(profile, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
