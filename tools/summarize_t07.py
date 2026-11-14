"""Collect reproducible run evidence; never turn detection counts into quality metrics."""
import json
from pathlib import Path


def main():
    runs = {}
    for name in ("baseline", "poc", "blur_all"):
        folder = Path("out") / name
        info = json.loads((folder / "run_info.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "detection_summary.json").read_text(encoding="utf-8"))
        if info["status"] != 0 or info["track"] != "t07":
            raise ValueError(f"Not a successful T07 run: {folder}")
        n = info["stats"]["n_items"]
        runs[name] = {"status": info["status"], "seed": info["seed"], "offline": info["offline"],
                      "n_images": n, "n_regions": summary["n_regions"],
                      "runtime_seconds": info["runtime_seconds"],
                      "runtime_per_100_images_seconds": info["runtime_seconds"]*100/n,
                      "source": str(folder / "run_info.json"),
                      "detections_by_source": summary["detections_by_source"],
                      "quality_metrics": None}
    payload = {"runs": runs, "note": "Counts and runtime only; no precision/recall or official quality claim without GT. Runtime includes export and concurrent machine load."}
    Path("reports/t07_run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
