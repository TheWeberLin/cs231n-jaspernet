# -*- coding: utf-8 -*-
"""
Species classification for detections — DINOv2 test-time inference.

Crops each detection out of its source photo in memory (via crop.py) and
classifies it with a trained DINOv2 checkpoint, writing predictions back into
the detections CSV. No crop files are ever written.

Config (MODEL_NAME / NUM_CLASSES / IMG_SIZE) must stay in sync with training
(DINO_layerwise).

Usage:
    from classify import load_model, classify_detections_csv
    model = load_model("dino_v2_..._wildlife.pth")
    df = classify_detections_csv(
        "detections_out.csv",
        "detections_with_species.csv",
        photo_root="jasper-wildlife/ct_photos",
        model=model,
    )

CLI:
    python classify.py detections_out.csv out.csv \
        --checkpoint dino_v2_..._wildlife.pth \
        --photo-root jasper-wildlife/ct_photos
"""

import os
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import timm

from crop import build_photo_path, crop_detection_row, PHOTO_ROOT, DEFAULT_CONTEXT

labels = pd.read_csv('species_to_idx.csv', index_col=0)
CLASS_NAMES = list(labels.columns)

# ----------------------------------------------------------------------------
# Config — keep in sync with training (DINO_layerwise).
# ----------------------------------------------------------------------------
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
NUM_CLASSES = 35
IMG_SIZE = 224


# ----------------------------------------------------------------------------
# Model loading / inference
# ----------------------------------------------------------------------------
def pick_device(prefer="auto"):
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device=None):
    """Build DINOv2 and load local .pth weights (full checkpoint or bare state_dict)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    device = pick_device() if device is None else device
    print(f"Loading classifier weights from {checkpoint_path} onto {device}")

    model = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, img_size=IMG_SIZE
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def build_transform(model):
    data_config = timm.data.resolve_model_data_config(model=model, args={"img_size": IMG_SIZE})
    return timm.data.create_transform(**data_config, is_training=False)


def _label_for(class_id):
    if CLASS_NAMES and 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return class_id


@torch.no_grad()
def classify_detections_csv(
    in_csv_path,
    out_csv_path,
    photo_root=PHOTO_ROOT,
    model=None,
    checkpoint_path=None,
    device=None,
    context=DEFAULT_CONTEXT,
    multiple_col="n_detections",
    batch_size=64,
    top_k=5,
    add_confidence=True,
    add_topk=False,
    species_col="pred_species",
) -> pd.DataFrame:
    """
    Crop each detection out of its source photo in memory and classify it,
    writing predictions back into the CSV. No crop files are created.

    Expects a detections CSV like megadetector_pipeline.py produces:
      CameraPath, Name, img_w, img_h, bbox_x, bbox_y, bbox_w, bbox_h, multiple

    Cropping/branching (bbox vs. whole-image second check, ERROR rows) is
    delegated to crop.crop_detection_row. Rows with a missing photo or no
    usable bbox are skipped.
    """
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Provide either `model` or `checkpoint_path`.")
        model = load_model(checkpoint_path, device=device)

    device = next(model.parameters()).device
    transform = build_transform(model)

    df = pd.read_csv(in_csv_path)
    n = len(df)

    pred_ids = [None] * n
    pred_labels = [None] * n
    pred_confs = [None] * n
    pred_topk = [None] * n

    batch_rows = []
    batch_tensors = []

    def flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        probs = F.softmax(model(batch), dim=1)
        k = min(top_k, probs.shape[1])
        tp, ti = torch.topk(probs, k=k, dim=1)
        tp, ti = tp.cpu().numpy(), ti.cpu().numpy()
        for j, ridx in enumerate(batch_rows):
            cid = int(ti[j][0])
            pred_ids[ridx] = cid
            pred_labels[ridx] = _label_for(cid)
            pred_confs[ridx] = float(tp[j][0])
            if add_topk:
                pred_topk[ridx] = [
                    {"class_id": int(ti[j][m]), "probability": float(tp[j][m])}
                    for m in range(k)
                ]
        batch_rows.clear()
        batch_tensors.clear()

    # Cache opened photos so multiple detections in one image don't reopen it.
    _photo_cache = {"path": None, "img": None}

    def get_photo(path):
        if _photo_cache["path"] != path:
            _photo_cache["img"] = Image.open(path).convert("RGB")
            _photo_cache["path"] = path
        return _photo_cache["img"]

    print(f"Cropping + classifying {n} rows on {device} (no crop files written)...")
    for i, row in df.iterrows():
        try:
            path = build_photo_path(row, photo_root)
            if not os.path.exists(path):
                print(f"[{i + 1}/{n}] MISSING photo: {path}")
                continue
            photo = get_photo(path)

            crop = crop_detection_row(
                photo, row, context=context, multiple_col=multiple_col
            )
            if crop is None:
                print(f"[{i + 1}/{n}] SKIP: detection flagged but bbox missing")
                continue

            batch_tensors.append(transform(crop))
            batch_rows.append(i)
        except Exception as e:
            print(f"[{i + 1}/{n}] FAILED row: {e}")
            continue

        if len(batch_tensors) >= batch_size:
            flush()

    flush()

    df[species_col] = pred_ids
    if CLASS_NAMES:
        df[f"{species_col}_name"] = pred_labels
    if add_confidence:
        df[f"{species_col}_conf"] = pred_confs
    if add_topk:
        df[f"{species_col}_topk"] = pred_topk

    df.to_csv(out_csv_path, index=False)
    classified = sum(1 for v in pred_ids if v is not None)
    print(f"\nSaved {classified}/{n} classified rows -> {out_csv_path}")
    return df


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Crop detections in memory and classify species, writing to a CSV. No crop files."
    )
    p.add_argument("input_csv", help="Detections CSV (CameraPath, Name, bbox_*, img_w/h, ...)")
    p.add_argument("output_csv", help="Where to write the augmented CSV")
    p.add_argument("--checkpoint", required=True, help="Path to trained .pth checkpoint")
    p.add_argument("--photo-root", default=PHOTO_ROOT, help="Root CameraPath is relative to")
    p.add_argument("--context", type=float, default=DEFAULT_CONTEXT, help="Bbox padding fraction")
    p.add_argument("--species-col", default="predicted_species")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--multiple-col", default="n_detections", help="Column with total detection count")
    p.add_argument("--add-topk", action="store_true")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    mdl = load_model(args.checkpoint, device=pick_device(args.device))
    classify_detections_csv(
        args.input_csv,
        args.output_csv,
        photo_root=args.photo_root,
        model=mdl,
        context=args.context,
        multiple_col=args.multiple_col,
        batch_size=args.batch_size,
        top_k=args.top_k,
        species_col=args.species_col,
        add_topk=args.add_topk,
    )
