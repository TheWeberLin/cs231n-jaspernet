# -*- coding: utf-8 -*-
"""
Visual check for bounding boxes — draws each detection's bbox on the FULL
source photo (no cropping) and saves it so you can eyeball whether the boxes
line up with the animals.

Unlike test_crop.py, this doesn't crop or square anything; it just overlays the
normalized MegaDetector bbox on the original frame. Multiple detections in the
same photo are drawn on the same image.

Usage:
    python test_bbox.py detections_out.csv \
        --photo-root jasper-wildlife/ct_photos \
        --out-dir bbox_check \
        --limit 40
"""

import os
import argparse

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from crop import build_photo_path, row_has_detection, PHOTO_ROOT


def _load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_bbox(draw, img_w, img_h, x, y, w, h, label, color=(255, 60, 60), font=None):
    """Draw one normalized bbox (x,y,w,h as fractions) onto the image."""
    px1 = x * img_w
    py1 = y * img_h
    px2 = (x + w) * img_w
    py2 = (y + h) * img_h

    line_w = max(2, int(round(min(img_w, img_h) * 0.004)))
    draw.rectangle([px1, py1, px2, py2], outline=color, width=line_w)

    if label:
        tx, ty = px1 + 2, max(0, py1 - (font.size + 4 if font else 14))
        try:
            bbox = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle(bbox, fill=color)
        except Exception:
            pass
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)


def main():
    p = argparse.ArgumentParser(description="Draw detection bboxes on full images for inspection.")
    p.add_argument("input_csv", help="Detections CSV (CameraPath, Name, bbox_*, multiple, ...)")
    p.add_argument("--photo-root", default=PHOTO_ROOT, help="Root CameraPath is relative to")
    p.add_argument("--out-dir", default="bbox_check", help="Where to write annotated images")
    p.add_argument("--multiple-col", default="multiple")
    p.add_argument("--limit", type=int, default=50, help="Max photos to annotate (0 = all)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    # Group rows by source photo so all boxes for one image land on one output.
    df["_path"] = df.apply(lambda r: _safe_path(r, args.photo_root), axis=1)
    groups = [(path, g) for path, g in df.groupby("_path", sort=False) if path]

    if args.limit and args.limit > 0:
        groups = groups[: args.limit]

    saved = 0
    print(f"Annotating up to {len(groups)} photos -> {args.out_dir}/")
    for path, g in groups:
        try:
            if not os.path.exists(path):
                print(f"MISSING photo: {path}")
                continue

            img = Image.open(path).convert("RGB")
            img_w, img_h = img.size
            draw = ImageDraw.Draw(img)
            font = _load_font(max(12, int(min(img_w, img_h) * 0.03)))

            n_boxes = 0
            for i, row in g.iterrows():
                if not row_has_detection(row, args.multiple_col):
                    continue  # no detection -> nothing to draw
                bx = row.get("bbox_x")
                if bx is None or (isinstance(bx, float) and pd.isna(bx)):
                    continue  # ERROR row, no usable bbox

                conf = row.get("conf")
                cat = row.get("category")
                label_bits = []
                if cat is not None and not (isinstance(cat, float) and pd.isna(cat)):
                    label_bits.append(str(cat))
                if conf is not None and not (isinstance(conf, float) and pd.isna(conf)):
                    try:
                        label_bits.append(f"{float(conf):.2f}")
                    except Exception:
                        pass
                label = " ".join(label_bits)

                draw_bbox(
                    draw, img_w, img_h,
                    float(row["bbox_x"]), float(row["bbox_y"]),
                    float(row["bbox_w"]), float(row["bbox_h"]),
                    label, font=font,
                )
                n_boxes += 1

            name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(args.out_dir, f"{name}_{n_boxes}box.jpg")
            img.save(out_path, "JPEG", quality=90)
            saved += 1
            print(f"{n_boxes} box(es) -> {out_path}")
        except Exception as e:
            print(f"FAILED {path}: {e}")
            continue

    print(f"\nSaved {saved} annotated photos to {args.out_dir}/")


def _safe_path(row, photo_root):
    try:
        return build_photo_path(row, photo_root)
    except Exception:
        return None


if __name__ == "__main__":
    main()
