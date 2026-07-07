"""
full_pipeline.py — MegaDetector detection followed by DINOv2 species
classification, driven by a metadata CSV export from MySQL.

    metadata CSV --[detect]--> detections CSV --[classify]--> species CSV
"""
import argparse
from agg_files import collapse_to_image_rows
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
    parser.add_argument("--classifier", default="DINO.pth", help="Path to trained DINOv2 .pth")
    parser.add_argument("--photo-root", default=PHOTO_ROOT, help="Root CameraPath is relative to")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR_PATH, help="MegaDetector version")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--batch-size", type=int, default=64)
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
    classifier = load_classifier(args.classifier, device=pick_device(args.device))
    df =classify_detections_csv(
        args.detections_csv,
        args.species_csv,
        photo_root=args.photo_root,
        model=classifier,
        batch_size=args.batch_size,
        add_topk=True,
    )

    print(f"\nDone. Detections -> {args.detections_csv}, species -> {args.species_csv}")
    
    collapsed = collapse_to_image_rows(df, species_col="pred_species")
    collapsed.to_csv(args.species_csv, index=False)

    print("Collapsed to one row per image, with lists of detections per image.")