from typing import Tuple

import numpy as np
from PIL import Image

from config.models import ModelConfig
from ml.segmentation import perform_segmentation


_INPUT_SIZE = (512, 512)


def create_overlay_image(
    original_image: Image.Image,
    mask_image: Image.Image,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    orig = np.array(original_image.convert("RGB")).astype(np.float32)
    mask = np.array(mask_image).astype(np.float32)

    if mask.max() > 1.0:
        mask = mask / 255.0

    color_array = np.zeros_like(orig)
    color_array[:, :, 0] = color[0]
    color_array[:, :, 1] = color[1]
    color_array[:, :, 2] = color[2]

    blended = orig.copy()
    for c in range(3):
        blended[:, :, c] = (
            mask * (alpha * color_array[:, :, c] + (1 - alpha) * orig[:, :, c])
            + (1 - mask) * orig[:, :, c]
        )

    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def generate_mask(tile_path: str, model_config: ModelConfig) -> np.ndarray:
    mask_path = perform_segmentation(tile_path, model_config.api_model_name, model_config.weights_path)
    mask_img = Image.open(mask_path).convert("L")
    return (np.array(mask_img) > 0).astype(np.uint8)
