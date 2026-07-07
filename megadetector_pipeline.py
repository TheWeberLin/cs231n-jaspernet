
import argparse

import pandas as pd
from PIL import Image
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from megadetector.detection import run_detector

CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}


def load_model(model_path):
    """Load the MegaDetector model once, reuse across batches."""
    print(f"Loading MegaDetector model: {model_path}")
    return run_detector.load_detector(model_path)


def build_image_path(camera_path: str, name: str, photo_root) -> str:
    """Reconstruct the local file path from CameraPath + filename."""
    return f"{photo_root}/{camera_path}/{name}"


def run_detections_from_csv(
    meta_csv_path: str,
    out_csv_path: str,
    photo_root,
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
                        "rough_category": CATEGORIES.get(d["category"], d["category"]),
                        "detection_conf": round(d["conf"], 4),
                        "img_w": image.size[0],
                        "img_h": image.size[1],
                        "bbox_x": d["bbox"][0],
                        "bbox_y": d["bbox"][1],
                        "bbox_w": d["bbox"][2],
                        "bbox_h": d["bbox"][3],
                        "n_detections": d_cnt,
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
                        "rough_category": None,
                        "detection_conf": None,
                        "img_w": image.size[0],
                        "img_h": image.size[1],
                        "bbox_x": None,
                        "bbox_y": None,
                        "bbox_w": None,
                        "bbox_h": None,
                        "n_detections": 0,
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


# def run_in_chunks(
#     meta_csv_path: str,
#     out_prefix: str,
#     chunk_size: int = 50,
#     photo_root,
#     conf_threshold: float = 0.2,
#     model=None,
# ) -> pd.DataFrame:
#     """
#     Same as run_detections_from_csv, but processes the metadata CSV in chunks
#     and writes one output CSV per chunk (useful for long runs / crash recovery),
#     then returns the combined DataFrame.
#     """
#     if model is None:
#         model = load_model()

#     meta_df = pd.read_csv(meta_csv_path)
#     all_dfs = []

#     for start in range(0, len(meta_df), chunk_size):
#         chunk = meta_df.iloc[start:start + chunk_size]
#         chunk_csv = f"{out_prefix}_chunk_{start}-{start + len(chunk) - 1}.csv"
#         tmp_input = "_tmp_chunk_input.csv"
#         chunk.to_csv(tmp_input, index=False)
#         df = run_detections_from_csv(
#             tmp_input,
#             chunk_csv,
#             photo_root=photo_root,
#             conf_threshold=conf_threshold,
#             model=model,
#         )
#         all_dfs.append(df)

#     return pd.concat(all_dfs, ignore_index=True)

