# -*- coding: utf-8 -*-
"""
MegaDetector pipeline driven by a metadata CSV export from MySQL.

Local filesystem version — photos already live on disk under the same
folder structure that used to live in the GCS bucket:

    jasper-wildlife/ct_photos/<CameraPath>/<Name>
    e.g. jasper-wildlife/ct_photos/CAMERA_ARRAY_A/camera_A2/p_000665.jpg

Workflow:
  1. You export new (unlabeled) photo metadata from MySQL to a CSV.
     Required columns: MetaId, Name, CameraPath
       - MetaId      : the primary key from your metadata table (used to join back)
       - Name        : image filename, e.g. "p_000665.jpg"
       - CameraPath  : e.g. "CAMERA_ARRAY_A/camera_A2" (folder the photo lives in,
                        relative to PHOTO_ROOT)
  2. This script opens each photo from disk, runs MegaDetector on it, and
     writes one row per detection (plus MetaId) to an output CSV.
  3. You import that CSV back into MySQL, joining on MetaId.

Row format mirrors your original script
(Name, CameraPath, category, conf, img_w, img_h, bbox_x/y/w/h, multiple),
with MetaId added as the join key back to your metadata table.

Usage:
    python megadetector_pipeline.py new_photos_metadata.csv detections_out.csv

    # or import and call directly:
    from megadetector_pipeline import load_model, run_detections_from_csv
    model = load_model()
    df = run_detections_from_csv("new_photos_metadata.csv", "detections_out.csv", model=model)
"""

import argparse

import pandas as pd
from PIL import Image
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from megadetector.detection import run_detector

# Root folder on disk that CameraPath is relative to.
PHOTO_ROOT = "jasper-wildlife/ct_photos"

CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}

# MDV5A / MDV5B are the current MegaDetector model aliases; run_detector will
# resolve/download the right weights. Swap for a specific local .pt path if
# you're pinning a version.
MODEL_PATH = "MDV5A"


def load_model(model_path: str = MODEL_PATH):
    """Load the MegaDetector model once, reuse across batches."""
    print(f"Loading MegaDetector model: {model_path}")
    return run_detector.load_detector(model_path)


def build_image_path(camera_path: str, name: str, photo_root: str = PHOTO_ROOT) -> str:
    """Reconstruct the local file path from CameraPath + filename."""
    return f"{photo_root}/{camera_path}/{name}"


def run_detections_from_csv(
    meta_csv_path: str,
    out_csv_path: str,
    photo_root: str = PHOTO_ROOT,
    conf_threshold: float = 0.2,
    model=None,
) -> pd.DataFrame:
    """
    Read a metadata CSV (MetaId, Name, CameraPath, ...), run MegaDetector on
    each referenced photo (read from local disk), and write a detections CSV
    joinable via MetaId.
    """
    if model is None:
        model = load_model()

    meta_df = pd.read_csv(meta_csv_path)

    required_cols = {"MetaId", "Name", "CameraPath"}
    missing = required_cols - set(meta_df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing required columns: {missing}")

    rows = []
    n = len(meta_df)

    for i, meta_row in meta_df.iterrows():
        meta_id = meta_row["MetaId"]
        name = meta_row["Name"]
        camera_path = meta_row["CameraPath"]
        image_path = build_image_path(camera_path, name, photo_root)

        try:
            image = Image.open(image_path).convert("RGB")

            result = model.generate_detections_one_image(image)
            detections = [d for d in result["detections"] if d["conf"] > conf_threshold]
            d_cnt = len(detections)

            if detections:
                print(f"[{i + 1}/{n}] {name} — {d_cnt} detection(s)")
                out_rows = [
                    {
                        "MetaId": meta_id,
                        "Name": name,
                        "CameraPath": camera_path,
                        "category": CATEGORIES.get(d["category"], d["category"]),
                        "conf": round(d["conf"], 4),
                        "img_w": image.size[0],
                        "img_h": image.size[1],
                        "bbox_x": d["bbox"][0],
                        "bbox_y": d["bbox"][1],
                        "bbox_w": d["bbox"][2],
                        "bbox_h": d["bbox"][3],
                        "multiple": d_cnt > 1,
                    }
                    for d in detections
                ]
            else:
                print(f"[{i + 1}/{n}] {name} — 0 detections")
                out_rows = [
                    {
                        "MetaId": meta_id,
                        "Name": name,
                        "CameraPath": camera_path,
                        "category": None,
                        "conf": None,
                        "img_w": image.size[0],
                        "img_h": image.size[1],
                        "bbox_x": None,
                        "bbox_y": None,
                        "bbox_w": None,
                        "bbox_h": None,
                        "multiple": None,
                    }
                ]

            rows.extend(out_rows)

        except Exception as e:
            print(f"[{i + 1}/{n}] FAILED {image_path}: {e}")
            rows.append(
                {
                    "MetaId": meta_id,
                    "Name": name,
                    "CameraPath": camera_path,
                    "category": "ERROR",
                    "conf": None,
                    "img_w": None,
                    "img_h": None,
                    "bbox_x": None,
                    "bbox_y": None,
                    "bbox_w": None,
                    "bbox_h": None,
                    "multiple": None,
                }
            )
            continue

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv_path, index=False)
    print(f"\nSaved {len(out_df)} detection rows for {n} photos → {out_csv_path}")

    return out_df


def run_in_chunks(
    meta_csv_path: str,
    out_prefix: str,
    chunk_size: int = 50,
    photo_root: str = PHOTO_ROOT,
    conf_threshold: float = 0.2,
    model=None,
) -> pd.DataFrame:
    """
    Same as run_detections_from_csv, but processes the metadata CSV in chunks
    and writes one output CSV per chunk (useful for long runs / crash recovery),
    then returns the combined DataFrame.
    """
    if model is None:
        model = load_model()

    meta_df = pd.read_csv(meta_csv_path)
    all_dfs = []

    for start in range(0, len(meta_df), chunk_size):
        chunk = meta_df.iloc[start:start + chunk_size]
        chunk_csv = f"{out_prefix}_chunk_{start}-{start + len(chunk) - 1}.csv"
        tmp_input = "_tmp_chunk_input.csv"
        chunk.to_csv(tmp_input, index=False)
        df = run_detections_from_csv(
            tmp_input,
            chunk_csv,
            photo_root=photo_root,
            conf_threshold=conf_threshold,
            model=model,
        )
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MegaDetector on photos listed in a metadata CSV (local files)."
    )
    parser.add_argument("metadata_csv", help="Path to input metadata CSV (MetaId, Name, CameraPath, ...)")
    parser.add_argument("output_csv", help="Path to write detection results CSV")
    parser.add_argument("--photo-root", default=PHOTO_ROOT, help="Root folder that CameraPath is relative to")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--model", default=MODEL_PATH, help="MegaDetector model version/path")
    args = parser.parse_args()

    mdl = load_model(args.model)
    run_detections_from_csv(
        args.metadata_csv,
        args.output_csv,
        photo_root=args.photo_root,
        conf_threshold=args.conf,
        model=mdl,
    )
