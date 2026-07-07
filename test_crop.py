# -*- coding: utf-8 -*-
"""
Visual check for crop.py — crops each detection out of its source photo and
saves the result to disk so you can eyeball whether the cropping is right.

This writes files ONLY for inspection; it's throwaway. The real pipeline keeps
crops in memory.

Usage:
    python test_crop.py detections_out.csv \
        --photo-root jasper-wildlife/ct_photos \
        --out-dir crop_check \
        --limit 40
"""

import os
import argparse

import pandas as pd
from PIL import Image

from crop import build_photo_path, crop_detection_row, row_has_detection, PHOTO_ROOT, DEFAULT_CONTEXT


def main():
    p = argparse.ArgumentParser(description="Crop detections and save them for visual inspection.")
    p.add_argument("input_csv", help="Detections CSV (CameraPath, Name, bbox_*, multiple, ...)")
    p.add_argument("--photo-root", default=PHOTO_ROOT, help="Root CameraPath is relative to")
    p.add_argument("--out-dir", default="crop_check", help="Where to write crop images")
    p.add_argument("--context", type=float, default=DEFAULT_CONTEXT, help="Bbox padding fraction")
    p.add_argument("--multiple-col", default="multiple")
    p.add_argument("--limit", type=int, default=50, help="Max rows to crop (0 = all)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    n = len(df)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    # Cache opened photos so multiple detections in one image don't reopen it.
    _cache = {"path": None, "img": None}

    def get_photo(path):
        if _cache["path"] != path:
            _cache["img"] = Image.open(path).convert("RGB")
            _cache["path"] = path
        return _cache["img"]

    saved = 0
    print(f"Cropping up to {len(df)} of {n} rows -> {args.out_dir}/")
    for i, row in df.iterrows():
        try:
            path = build_photo_path(row, args.photo_root)
            if not os.path.exists(path):
                print(f"[{i}] MISSING photo: {path}")
                continue

            photo = get_photo(path)
            crop = crop_detection_row(
                photo, row, context=args.context, multiple_col=args.multiple_col
            )
            if crop is None:
                print(f"[{i}] SKIP: detection flagged but bbox missing")
                continue

            det = "det" if row_has_detection(row, args.multiple_col) else "full"
            name = os.path.splitext(str(row.get("Name", f"row{i}")))[0]
            out_path = os.path.join(args.out_dir, f"{i:05d}_{name}_{det}_{crop.size[0]}x{crop.size[1]}.jpg")
            crop.save(out_path, "JPEG", quality=92)
            saved += 1
            print(f"[{i}] {det} crop {crop.size} -> {out_path}")
        except Exception as e:
            print(f"[{i}] FAILED: {e}")
            continue

    print(f"\nSaved {saved} crops to {args.out_dir}/")


if __name__ == "__main__":
    main()
