"""Analytic checks for reviewed-reference experiment denominators and coverage."""
import pandas as pd
import pytest

from tools.evaluate_t07_reviewed import score


def reference():
    return pd.DataFrame([
        {"image_id": "a", "region_id": "a0", "x": 0, "y": 0, "w": 10, "h": 10,
         "type": "face", "must_redact": 1},
        {"image_id": "b", "region_id": "b0", "x": 0, "y": 0, "w": 10, "h": 10,
         "type": "plate", "must_redact": 1},
    ])


def test_union_coverage_at_ninety_percent_passes_and_empty_image_is_missed():
    # Different prediction types still physically cover PII: primary is mask coverage.
    predictions = {"a": [dict(x=0, y=0, w=5, h=10, type="plate"),
                          dict(x=4, y=0, w=5, h=10, type="plate")], "b": []}
    result, regions = score(reference(), predictions, ["a", "b"], {"a": (20, 20), "b": (20, 20)})
    assert result["n_reference_regions"] == 2
    assert result["missed_regions"] == 1
    assert result["pii_miss_rate"] == 0.5
    assert result["region_success_rate"] == 0.5
    assert list(regions.coverage) == [0.9, 0.0]
    assert result["by_type"]["face"]["missed"] == 0
    assert result["by_type"]["plate"]["missed"] == 1
    assert result["over_redaction_rate"] == 0


def test_only_requested_reviewed_images_contribute_to_denominators():
    predictions = {"a": [dict(x=0, y=0, w=20, h=10, type="face")], "b": []}
    result, _ = score(reference(), predictions, ["a"], {"a": (20, 20), "b": (20, 20)})
    assert result["n_images"] == 1
    assert result["n_reference_regions"] == 1
    assert result["region_success_rate"] == 1
    assert result["over_redaction_rate"] == pytest.approx(0.5)
    assert not result["guardrail_pass"]
    assert result["bootstrap_95ci_miss"] == [0, 0]
    assert result["bootstrap_95ci_over"] == [0.5, 0.5]
