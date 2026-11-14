"""Router contract checks on generated toy pixels, not measured photo accuracy."""
import cv2
import numpy as np
import pytest

from src.t07_router import RouterConfig, detect_routed, route_image


def drawing():
    image = np.full((120, 180, 3), (160, 190, 210), np.uint8)
    cv2.rectangle(image, (30, 30), (85, 85), (30, 80, 140), -1)
    return image


def test_flat_drawing_and_noise_route_from_pixels():
    assert route_image(drawing())["route"] == "2d"
    noise = np.random.default_rng(42).integers(0, 256, (120, 180, 3), dtype=np.uint8)
    assert route_image(noise)["route"] == "real"
    # Noise is only a texture toy; this does not validate real photographs.


def test_outer_black_border_does_not_change_content_features():
    image = drawing()
    bordered = cv2.copyMakeBorder(image, 12, 12, 15, 15, cv2.BORDER_CONSTANT, value=0)
    plain, padded = route_image(image), route_image(bordered)
    assert padded["features"]["content_roi_xywh"] == [15, 12, 180, 120]
    assert plain["features"]["flat_fraction"] == padded["features"]["flat_fraction"]
    assert plain["route"] == padded["route"]


def test_internal_black_content_is_retained():
    image = drawing()
    image[40:70, 110:145] = 0
    features = route_image(image)["features"]
    assert features["content_roi_xywh"] == [0, 0, 180, 120]
    assert features["quantized_unique_colors"] == 3


def test_blank_and_tiny_inputs_are_uncertain():
    for shape in [(120, 180, 3), (1, 1, 3)]:
        result = route_image(np.zeros(shape, np.uint8))
        assert result["route"] == "both"
        assert result["reason"] == "low_information"
        assert result["calibration_status"] == "uncalibrated"


def test_near_threshold_routes_both():
    image = drawing()
    measured = route_image(image)["features"]["flat_fraction"]
    config = RouterConfig(flat_threshold=measured, uncertainty_margin=0.001)
    assert route_image(image, config)["route"] == "both"


def test_dispatch_skips_real_detector_for_drawing_without_mutating_image():
    image = drawing()
    original = image.copy()

    def real(_):
        raise AssertionError("Real branch must not run for this drawing")

    def synthetic(pixels):
        pixels[:] = 0
        return [], 2

    result = detect_routed(image, real_detector=real, synthetic_detector=synthetic)
    assert result["ambiguous_heads"] == 2
    assert result["requires_review"]
    np.testing.assert_array_equal(image, original)


def test_union_retains_larger_plate_tail_and_separate_face():
    image = np.full((80, 140, 3), 100, np.uint8)  # low information -> both
    def box(x, width, kind="plate", confidence=0.8):
        return {"x": x, "y": 20, "w": width, "h": 12, "type": kind,
                "confidence": confidence, "source": kind}

    result = detect_routed(
        image, real_detector=lambda _: [box(40, 25, confidence=0.99)],
        synthetic_detector=lambda _: ([box(30, 50), box(40, 25, "face")], 0),
    )
    plates = [b for b in result["boxes"] if b["type"] == "plate"]
    assert len(plates) == 1
    assert plates[0]["x"] == 30 and plates[0]["w"] == 50
    assert len(result["boxes"]) == 2
    assert result["routing"]["route"] == "both"
    assert result["requires_review"]


@pytest.mark.parametrize("image", [None, np.zeros((10, 10), np.uint8),
                                   np.zeros((10, 10, 3), np.float32)])
def test_invalid_images_fail_explicitly(image):
    with pytest.raises(ValueError):
        route_image(image)
