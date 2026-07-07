"""
full_pipeline.py — MegaDetector detection followed by DINOv2 species
classification, driven by a metadata CSV export from MySQL.

    metadata CSV --[detect]--> detections CSV --[classify]--> species CSV
"""
import argparse

from megadetector_pipeline import run_detections_from_csv, load_model as load_detector
from classify import classify_detections_csv, load_model as load_classifier, pick_device

PHOTO_ROOT = "jasper-wildlife/ct_photos"
DEFAULT_DETECTOR_PATH = "MDV5A"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect animals then classify species; writes one CSV per stage."
    )
    parser.add_argument("input_csv", help="Input metadata CSV (MetaId, Name, CameraPath, ...)")
    parser.add_argument("detections_csv", help="Where to write MegaDetector detections")
    parser.add_argument("species_csv", help="Where to write the classified output")
    parser.add_argument("--checkpoint", required=True, help="Path to trained DINOv2 .pth")
    parser.add_argument("--photo-root", default=PHOTO_ROOT, help="Root CameraPath is relative to")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR_PATH, help="MegaDetector version")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--add-topk", action="store_true")
    args = parser.parse_args()

    # Stage 1: detection
    detector = load_detector(args.detector)
    run_detections_from_csv(
        args.input_csv,
        args.detections_csv,
        photo_root=args.photo_root,
        model=detector,
    )

    # Stage 2: classification (crops in memory from the detections CSV)
    classifier = load_classifier(args.checkpoint, device=pick_device(args.device))
    classify_detections_csv(
        args.detections_csv,
        args.species_csv,
        photo_root=args.photo_root,
        model=classifier,
        batch_size=args.batch_size,
        add_topk=args.add_topk,
    )

    print(f"\nDone. Detections -> {args.detections_csv}, species -> {args.species_csv}")
# """
# MegaDetector pipeline driven by a metadata CSV export from MySQL.

# Local filesystem version — photos already live on disk under the same
# folder structure that used to live in the GCS bucket:

#     jasper-wildlife/ct_photos/<CameraPath>/<Name>
#     e.g. jasper-wildlife/ct_photos/CAMERA_ARRAY_A/camera_A2/p_000665.jpg

# Workflow:
#   1. You export new (unlabeled) photo metadata from MySQL to a CSV.
#      Required columns: MetaId, Name, CameraPath
#        - MetaId      : the primary key from your metadata table (used to join back)
#        - Name        : image filename, e.g. "p_000665.jpg"
#        - CameraPath  : e.g. "CAMERA_ARRAY_A/camera_A2" (folder the photo lives in,
#                         relative to PHOTO_ROOT)
#   2. This script opens each photo from disk, runs MegaDetector on it, and
#      writes one row per detection (plus MetaId) to an output CSV.
#   3. You import that CSV back into MySQL, joining on MetaId.

# Row format mirrors your original script
# (Name, CameraPath, category, conf, img_w, img_h, bbox_x/y/w/h, multiple),
# with MetaId added as the join key back to your metadata table.

# Usage:
#     python megadetector_pipeline.py new_photos_metadata.csv detections_out.csv

#     # or import and call directly:
#     from megadetector_pipeline import load_model, run_detections_from_csv
#     model = load_model()
#     df = run_detections_from_csv("new_photos_metadata.csv", "detections_out.csv", model=model)
# """
# import argparse

# import pandas as pd
# import os
# from megadetector_pipeline import run_detections_from_csv, load_model
# from crop import crop_from_bbox, crop_detection_row

# # Root folder on disk that CameraPath is relative to.
# PHOTO_ROOT = "jasper-wildlife/ct_photos"


# # MDV5A / MDV5B are the current MegaDetector model aliases; run_detector will
# # resolve/download the right weights. Swap for a specific local .pt path if
# # you're pinning a version.
# DEFAULT_DETECTOR_PATH = "MDV5A"

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Run MegaDetector on photos listed in a metadata CSV (local files)."
#     )
#     parser.add_argument("input_csv", help="Path to input metadata CSV (MetaId, Name, CameraPath, ...)")
#     parser.add_argument("output_csv", help="Path to write results CSV")
#     parser.add_argument("--photo-root", default=PHOTO_ROOT, help="Root folder that CameraPath is relative to")
#     parser.add_argument('--model', default=DEFAULT_DETECTOR_PATH, help="which megadetector version to load")
#     args = parser.parse_args()

#     mdl = load_model(args.model)
#     detections_df = run_detections_from_csv(
#         args.input_csv,
#         args.output_csv,
#         photo_root=args.photo_root,
#         model=mdl,
#     )
#     crop = crop_from_bbox(pil_img, x, y, w, h, has_detection=True, context=0.15)
