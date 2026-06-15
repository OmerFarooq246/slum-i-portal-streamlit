import os
import numpy as np
import tensorflow as tf
from ml.ENB5_Segmenter import EfficientNetB5_Segmenter, EfficientNet_Segmenter_Config

TILE_SIZE = (512, 512, 3)
MASKS_FOLDER = "./public/outputs"


def perform_segmentation(img_path: str, model: str, weights_path: str) -> str:
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = resize_if_needed(img, TILE_SIZE[0], img_path)
        tile_names = None
        if img.shape != TILE_SIZE:
            tiles, tile_names = slice_into_patches(img)
        else:
            tiles = [img]

        if model in ["ENB5_Seg", "ENB5_Seg_PAK"]:
            masks = use_ENB5_Seg(tiles, model, weights_path)
        else:
            raise ValueError(f"Unsupported model: {model}")

        if tile_names:
            mask_merged = merge_masks(masks, tile_names)
        else:
            mask_merged = masks[0]

        name = img_path.split("/")[-1]
        name = name[:name.rindex(".")]
        output_path = f"{MASKS_FOLDER}/{model}"
        mask_path = f"{output_path}/{name}.png"
        os.makedirs(output_path, exist_ok=True)
        tf.keras.utils.save_img(mask_path, tf.expand_dims(mask_merged, axis=-1))
        return mask_path
    except Exception as e:
        raise RuntimeError(f"Error in perform_segmentation {model}: {str(e)}")


def merge_masks(masks, tile_names):
    try:
        if len(masks) != len(tile_names):
            raise ValueError(f"Number of masks ({len(masks)}) does not match number of tile names ({len(tile_names)}).")
        coords = []
        for name in tile_names:
            nums = tuple(map(int, name.split("/")[-1].split("_")[1:3]))
            coords.append(nums)

        widths  = [x + mask.shape[1] for mask, (x, _) in zip(masks, coords)]
        heights = [y + mask.shape[0] for mask, (_, y) in zip(masks, coords)]
        full_w = max(widths)
        full_h = max(heights)

        merged = np.zeros((full_h, full_w), dtype=np.uint8)
        for mask, (x, y) in zip(masks, coords):
            h, w = mask.shape
            merged[y:y+h, x:x+w] = np.maximum(merged[y:y+h, x:x+w], mask)
        return merged
    except Exception as e:
        raise RuntimeError(f"Error in merge_masks: {str(e)}")


def use_ENB5_Seg(tiles, model, weights_path: str):
    try:
        config = EfficientNet_Segmenter_Config()
        ENB5_Seg = EfficientNetB5_Segmenter(config)
        dummy_input = tf.zeros((1, *config.input_shape))
        _ = ENB5_Seg(dummy_input)
        ENB5_Seg.load_weights(weights_path)
        masks = []
        for tile in tiles:
            mask = ENB5_Seg.predict(tf.expand_dims(tf.cast(tile, tf.float32) / 255.0, axis=0))
            mask = tf.squeeze(mask, axis=0)
            mask = tf.argmax(mask, axis=-1)
            mask = tf.cast(mask * 255, tf.uint8)
            masks.append(mask)
        return masks
    except Exception as e:
        raise RuntimeError(f"Error in use_ENB5_Seg: {str(e)}")


def resize_if_needed(img: tf.Tensor, min_size: int, img_path: str) -> tf.Tensor:
    try:
        height, width, channels = img.shape
        if width < min_size or height < min_size:
            scale = max(min_size / width, min_size / height)
            new_width = int(round(width * scale))
            new_height = int(round(height * scale))
            img = tf.image.resize(img, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
            img = tf.cast(img, tf.uint8)
            tf.keras.utils.save_img(img_path, img.numpy())
    except Exception as e:
        raise RuntimeError(f"Error in resize_if_needed {img_path}: {str(e)}")
    return img


def slice_into_patches(raster):
    raster_w = raster.shape[1]
    raster_h = raster.shape[0]
    tiles = []
    tile_names = []
    patch_size = TILE_SIZE[0]

    for j in range(0, raster_h, patch_size):
        for i in range(0, raster_w, patch_size):
            i_temp = i
            j_temp = j
            if (raster_w - i) < patch_size:
                i_temp = raster_w - patch_size
            if (raster_h - j) < patch_size:
                j_temp = raster_h - patch_size
            tile_array = raster[j_temp:j_temp + patch_size, i_temp:i_temp + patch_size, :]
            tile_name = f"tile_{i_temp}_{j_temp}"
            tile_names.append(tile_name)
            tiles.append(tile_array)
    return tiles, tile_names
