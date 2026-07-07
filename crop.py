# -*- coding: utf-8 -*-
"""
In-memory cropping for detections — no intermediate crop files.

Ported from the Modal preprocessing script's padding / square-crop logic
(pad_to_square + the bbox expansion / square-ification in process_item), but
operating on in-memory PIL images so nothing is persisted.

Usage:
    from crop import crop_from_bbox, crop_detection_row
    crop = crop_from_bbox(pil_img, x, y, w, h, has_detection=True, context=0.15)
"""

import pandas as pd
from PIL import Image

DEFAULT_CONTEXT = 0.15  # same padding fraction as the Modal script
PHOTO_ROOT = "jasper-wildlife/ct_photos"


# ----------------------------------------------------------------------------
# Cropping (ported from the Modal preprocessing script, file-free)
# ----------------------------------------------------------------------------
def pad_to_square(img, background_color=(128, 128, 128)):
    """Pad a PIL image to a square by centering it on a gray canvas."""
    width, height = img.size
    if width == height:
        return img
    extra_space = abs(width - height)
    pad = extra_space // 2
    if width > height:
        new_size = (width, width)
        top = (0, pad)
    else:
        new_size = (height, height)
        top = (pad, 0)
    new_img = Image.new("RGB", new_size, background_color)
    new_img.paste(img, top)
    return new_img


def _center_square(img):
    """Center-crop a PIL image to a square on its shorter side (no padding)."""
    img_w, img_h = img.size
    min_dim = min(img_w, img_h)
    x1 = (img_w - min_dim) // 2
    y1 = (img_h - min_dim) // 2
    return img.crop((x1, y1, x1 + min_dim, y1 + min_dim))


def crop_from_bbox(
    img,
    x,
    y,
    w,
    h,
    has_detection=True,
    context=DEFAULT_CONTEXT,
):
    """
    Reproduce the crop behavior of process_item(), but on an in-memory PIL
    image and returning an in-memory PIL image (no disk I/O).

    img           : source PIL image (any mode; converted to RGB).
    x, y, w, h    : bbox as normalized fractions of width/height (what
                    MegaDetector emits). Ignored when has_detection=False.
    has_detection : maps to the `multiple` column. True (multiple != 0) means
                    MegaDetector found something -> scale + crop the bbox.
                    False (multiple == 0) means no detection -> feed the whole
                    image so the classifier acts as a second check. This mirrors
                    the original script, where the no-detection path fell through
                    to the full-frame branch: a center square crop.
    context       : fraction of box size to expand on each side before squaring.

    Returns a square RGB PIL image.
    """
    img = img.convert("RGB")
    img_w, img_h = img.size

    # No detection: use the whole image, center-cropped to a square.
    if not has_detection:
        return _center_square(img)

    # Detection present: bbox is normalized [0,1] -> convert to pixels.
    x = x * img_w
    y = y * img_h
    w = w * img_w
    h = h * img_h

    # A detection that already spans the whole frame -> center square crop.
    is_full_frame = (int(w) >= img_w - 1) and (int(h) >= img_h - 1)
    if is_full_frame:
        return _center_square(img)

    pad_w = w * context
    pad_h = h * context

    box_x1 = x - pad_w
    box_y1 = y - pad_h
    box_w = w + (2 * pad_w)
    box_h = h + (2 * pad_h)

    center_x = box_x1 + (box_w / 2)
    center_y = box_y1 + (box_h / 2)

    max_side = max(box_w, box_h)
    half_side = max_side / 2

    x1 = int(center_x - half_side)
    y1 = int(center_y - half_side)
    x2 = int(center_x + half_side)
    y2 = int(center_y + half_side)

    # Shift the square back inside the frame (same clamping as Modal script).
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
        x1 = max(0, x1)
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h
        y1 = max(0, y1)

    if (x2 - x1) > 0 and (y2 - y1) > 0:
        img = img.crop((x1, y1, x2, y2))

    # If the crop got clipped by the frame edge and is no longer square,
    # resize to square (matches Modal behavior).
    if img.size[0] != img.size[1]:
        target_size = max(img.size[0], img.size[1])
        img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)

    return img


# ----------------------------------------------------------------------------
# Path reconstruction + row-level convenience
# ----------------------------------------------------------------------------
def build_photo_path(row, photo_root=PHOTO_ROOT):
    """
    Reconstruct the source photo path from a detections-CSV row.
    Mirrors megadetector_pipeline.build_image_path: <photo_root>/<CameraPath>/<Name>.
    Falls back to a 'gcs_path' or 'source_path' column if present.
    """
    if "CameraPath" in row and "Name" in row and isinstance(row["CameraPath"], str):
        return f"{photo_root}/{row['CameraPath']}/{row['Name']}"
    for col in ("source_path", "gcs_path", "image_path"):
        if col in row and isinstance(row[col], str):
            p = row[col]
            if p.startswith("gs://"):
                p = p[5:]
            return p
    raise KeyError("Row lacks CameraPath+Name and any source_path/gcs_path column.")


def row_has_detection(row, multiple_col="multiple"):
    """
    `multiple` == 0 -> no MegaDetector detection -> classify whole image.
    Any nonzero (or missing) value -> real detection -> crop the bbox.
    """
    mult = row.get(multiple_col)
    return not (mult == 0 or (isinstance(mult, float) and mult == 0.0))


def crop_detection_row(photo, row, context=DEFAULT_CONTEXT, multiple_col="multiple"):
    """
    Crop a single detections-CSV row out of its (already-opened) source photo.
    Returns a square RGB PIL image, or None if the row is an ERROR row
    (flagged as a detection but missing a usable bbox).
    """
    has_detection = row_has_detection(row, multiple_col=multiple_col)

    if not has_detection:
        return crop_from_bbox(photo, 0, 0, 0, 0, has_detection=False)

    bx = row.get("bbox_x")
    if bx is None or (isinstance(bx, float) and pd.isna(bx)):
        return None  # detection flagged but no usable bbox

    return crop_from_bbox(
        photo,
        float(row["bbox_x"]),
        float(row["bbox_y"]),
        float(row["bbox_w"]),
        float(row["bbox_h"]),
        has_detection=True,
        context=context,
    )
