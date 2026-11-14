"""T07 baseline: local YuNet + YOLO default confidence, fixed blur, no human correction."""
from .t07_pipeline import run as run_t07


def run(ctx):
    run_t07(ctx, mode="baseline")
