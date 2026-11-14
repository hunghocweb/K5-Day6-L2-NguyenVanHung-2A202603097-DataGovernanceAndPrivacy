"""Trivial control required by T07; NOT an acceptable privacy/utility solution."""
from .t07_pipeline import run as run_t07


def run(ctx):
    run_t07(ctx, mode="blur_all")
