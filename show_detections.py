#!/usr/bin/env python3
"""
show_detections.py

Look up one photo in a camera-trap detection CSV (MegaDetector-style output)
and display every bounding box together with its species classification.

The CSV HAS A HEADER. Columns are found by name (not position), so they may be
reordered freely. Expected header names (logical meaning in parentheses):

    CameraPath          (camera path,               e.g. CAMERA_ARRAY_A/camera_A2)
    Name                (image file name,           e.g. quach_family.jpeg)
    bbox                ([[x, y, w, h], ...] normalized 0-1, origin top-left)
    MetaId              (internal record id)
    rough_category      (detector categories,       e.g. {'animal': 1})
    detection_conf      (per-box detection conf: float or [float, ...])
    img_w               (stored pixel width)
    img_h               (stored pixel height)
    n_detections        (number of detections)
    pred_species        (class-id counts,           e.g. {34: 1})
    pred_species_name   (class-name counts,         e.g. {'Wild turkey': 1})
    pred_species_conf   (top-class conf: float or [float, ...])
    pred_species_topk   (per-box top-k [{'class_id':.., 'probability':..}, ...])

Box i is index-aligned with detection_conf[i], pred_species_conf[i] and
pred_species_topk[i].

USAGE
-----
Text summary only:
    python show_detections.py --csv detections.csv \
        --camera-path CAMERA_ARRAY_A/camera_A2 --name quach_family.jpeg

Or with a single combined key (split on the last '/'):
    python show_detections.py --csv detections.csv \
        --key CAMERA_ARRAY_A/camera_A2/quach_family.jpeg

Also draw the boxes on the real photo (needs Pillow):
    python show_detections.py --csv detections.csv \
        --camera-path CAMERA_ARRAY_A/camera_A2 --name quach_family.jpeg \
        --images-root /path/to/photos --out annotated.jpg
"""

import argparse
import ast
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ----- logical field -> expected header name -------------------------------
# Columns are matched to headers by these names (case-insensitive, trimmed).
# Change the right-hand values if your header uses different labels.
COLUMNS = {
    "camera_path":       "CameraPath",
    "image_name":        "Name",
    "bboxes":            "bbox",
    "id":                "MetaId",
    "det_categories":    "rough_category",
    "det_conf":          "detection_conf",
    "width":             "img_w",
    "height":            "img_h",
    "num_detections":    "n_detections",
    "class_id_counts":   "pred_species",
    "class_name_counts": "pred_species_name",
    "top_class_conf":    "pred_species_conf",
    "classifications":   "pred_species_topk",
}
# The columns we cannot work without.
REQUIRED = ("camera_path", "image_name", "bboxes")

# A colour per class_id for drawing (cycled if more classes than colours).
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
]


def parse_literal(text):
    """ast.literal_eval a cell, returning the raw string if that fails."""
    if text is None:
        return None
    text = str(text).strip()
    if text == "":
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def as_list(value, n):
    """Normalize a cell that is either a scalar or a list into a length-n list."""
    if value is None:
        return [None] * n
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * n  # a scalar applies to the single detection


def normalize_classifications(raw, n):
    """
    Turn the classifications cell into a per-detection list of top-k lists.

    Single detection is stored as   [ {..}, {..}, ... ]    (top-k for one box)
    Multiple detections as          [ [ {..}, ... ], ... ]  (one top-k per box)
    """
    if not raw:
        return [[] for _ in range(n)]
    if isinstance(raw[0], dict):          # single detection, wrap it
        return [raw]
    return raw                            # already one entry per detection


def resolve_columns(fieldnames):
    """
    Map each logical field to the actual header present in the file.

    Matching is case-insensitive and ignores surrounding whitespace, so the
    physical column order does not matter. Returns (resolved, missing_required).
    """
    if not fieldnames:
        return {}, list(REQUIRED)
    lookup = {(fn or "").strip().lower(): fn for fn in fieldnames}
    resolved = {}
    for key, header in COLUMNS.items():
        actual = lookup.get(header.strip().lower())
        if actual is not None:
            resolved[key] = actual
    missing = [k for k in REQUIRED if k not in resolved]
    return resolved, missing


def cell(row, resolved, key, default=None):
    """Fetch a logical field from a DictReader row via the resolved header."""
    col = resolved.get(key)
    if col is None:
        return default
    return row.get(col, default)


def build_class_id_to_name(rows, resolved):
    """
    Derive a class_id -> species-name map from the count columns.

    class_id_counts and class_name_counts are aligned dicts, so we pair them
    positionally. Single-species rows are exact; for multi-species rows we
    trust dict ordering and take the most common mapping seen.
    """
    votes = defaultdict(Counter)
    for row in rows:
        ids = parse_literal(cell(row, resolved, "class_id_counts"))
        names = parse_literal(cell(row, resolved, "class_name_counts"))
        if not isinstance(ids, dict) or not isinstance(names, dict):
            continue
        for cid, name in zip(ids.keys(), names.keys()):
            votes[cid][name] += 1
    return {cid: ctr.most_common(1)[0][0] for cid, ctr in votes.items()}


def read_rows(csv_path):
    """Read the CSV with its header; return (rows, resolved_columns)."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        resolved, missing = resolve_columns(reader.fieldnames)
        if missing:
            want = ", ".join(COLUMNS[k] for k in missing)
            have = ", ".join(reader.fieldnames or [])
            sys.exit(f"CSV is missing required column(s): {want}\n"
                     f"Header found: {have}")
        rows = [row for row in reader]
    return rows, resolved


def find_matches(rows, resolved, camera_path, name):
    """Match on camera_path + name; fall back to name-only if camera_path is None."""
    out = []
    for row in rows:
        cp = (cell(row, resolved, "camera_path") or "").strip()
        nm = (cell(row, resolved, "image_name") or "").strip()
        if name is not None and nm != name.strip():
            continue
        if camera_path is not None and cp != camera_path.strip():
            continue
        out.append(row)
    return out


def summarize_row(row, resolved, id_to_name):
    """Parse one row into a tidy per-detection structure."""
    bboxes = parse_literal(cell(row, resolved, "bboxes")) or []
    n = len(bboxes)
    det_conf = as_list(parse_literal(cell(row, resolved, "det_conf")), n)
    classifications = normalize_classifications(
        parse_literal(cell(row, resolved, "classifications")), n
    )

    detections = []
    for i, box in enumerate(bboxes):
        topk = classifications[i] if i < len(classifications) else []
        top = topk[0] if topk else {}
        cid = top.get("class_id")
        detections.append({
            "index": i,
            "bbox": box,  # [x, y, w, h] normalized
            "det_conf": det_conf[i] if i < len(det_conf) else None,
            "class_id": cid,
            "class_name": id_to_name.get(cid, f"class_{cid}" if cid is not None else "?"),
            "class_conf": top.get("probability"),
            "topk": topk,
        })
    return {
        "camera_path": (cell(row, resolved, "camera_path") or "").strip(),
        "image_name": (cell(row, resolved, "image_name") or "").strip(),
        "id": (str(cell(row, resolved, "id") or "")).strip(),
        "width": parse_literal(cell(row, resolved, "width")),
        "height": parse_literal(cell(row, resolved, "height")),
        "num_detections": parse_literal(cell(row, resolved, "num_detections")),
        "det_categories": parse_literal(cell(row, resolved, "det_categories")),
        "class_name_counts": parse_literal(cell(row, resolved, "class_name_counts")),
        "detections": detections,
    }


def print_summary(info):
    print(f"\nImage         : {info['camera_path']}/{info['image_name']}")
    print(f"Record id     : {info['id']}")
    print(f"Stored size   : {info['width']} x {info['height']} px")
    print(f"Detections    : {info['num_detections']}")
    print(f"Det categories: {info['det_categories']}")
    print(f"Species counts: {info['class_name_counts']}")
    print()

    header = f"  {'#':>3}  {'det_conf':>8}  {'class (id)':<26}  {'cls_conf':>8}  bbox x,y,w,h (norm)"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for d in info["detections"]:
        dc = f"{d['det_conf']:.3f}" if isinstance(d["det_conf"], (int, float)) else "-"
        cc = f"{d['class_conf']:.4f}" if isinstance(d["class_conf"], (int, float)) else "-"
        label = f"{d['class_name']} ({d['class_id']})"
        box = d["bbox"]
        box_str = "[" + ", ".join(f"{v:.4f}" for v in box) + "]" if box else "-"
        print(f"  {d['index'] + 1:>3}  {dc:>8}  {label:<26}  {cc:>8}  {box_str}")
    print()


def draw_boxes(info, image_path, out_path):
    """Draw the boxes + labels on the real photo. Requires Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed - skipping image drawing "
              "(pip install pillow).", file=sys.stderr)
        return None

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    line_w = max(2, round(min(W, H) / 300))
    font_size = max(12, round(min(W, H) / 40))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    color_for = {}
    for d in info["detections"]:
        box = d["bbox"]
        if not box:
            continue
        x, y, w, h = box
        left, top = x * W, y * H
        right, bottom = (x + w) * W, (y + h) * H

        cid = d["class_id"]
        color = color_for.setdefault(cid, PALETTE[len(color_for) % len(PALETTE)])
        draw.rectangle([left, top, right, bottom], outline=color, width=line_w)

        cc = f" {d['class_conf']:.2f}" if isinstance(d["class_conf"], (int, float)) else ""
        label = f"{d['class_name']}{cc}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, top - th - 4)
        draw.rectangle([left, ly, left + tw + 6, ly + th + 4], fill=color)
        draw.text((left + 3, ly + 2), label, fill="black", font=font)

    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Show all bounding boxes + classifications for one camera-trap photo."
    )
    ap.add_argument("--csv", required=True, help="Path to the detections CSV.")
    ap.add_argument("--camera-path", help="e.g. CAMERA_ARRAY_A/camera_A2")
    ap.add_argument("--name", help="e.g. quach_family.jpeg")
    ap.add_argument("--key",
                    help="Combined 'camera_path/name'; split on the last '/'.")
    ap.add_argument("--folder-root",
                    help="Root dir of photos. Image looked up at "
                         "<root>/<camera_path>/<name>; enables drawing.")
    ap.add_argument("--image",
                    help="Explicit path to the photo (overrides --images-root).")
    ap.add_argument("--out", help="Where to save the annotated image.")
    args = ap.parse_args()

    camera_path, name = args.camera_path, args.name
    if args.key:
        if "/" in args.key:
            camera_path, name = args.key.rsplit("/", 1)
        else:
            name = args.key
    if name is None and camera_path is None:
        ap.error("Provide --name (and optionally --camera-path) or --key.")

    rows, resolved = read_rows(args.csv)
    if not rows:
        sys.exit("No data rows found in the CSV.")

    id_to_name = build_class_id_to_name(rows, resolved)
    matches = find_matches(rows, resolved, camera_path, name)

    if not matches:
        sys.exit(f"No match for camera_path={camera_path!r} name={name!r}.")
    if len(matches) > 1:
        print(f"Note: {len(matches)} rows matched; showing all.", file=sys.stderr)

    for row in matches:
        info = summarize_row(row, resolved, id_to_name)
        print_summary(info)

        # Locate the photo for drawing.
        image_path = None
        if args.image:
            image_path = Path(args.image)
        elif args.images_root:
            image_path = Path(args.images_root) / info["camera_path"] / info["image_name"]

        if image_path is not None:
            if image_path.exists():
                out = args.out or f"annotated_{info['image_name']}"
                if not out.lower().endswith((".jpg", ".jpeg", ".png")):
                    out += ".jpg"
                saved = draw_boxes(info, image_path, out)
                if saved:
                    print(f"Annotated image written to: {saved}\n")
            else:
                print(f"Photo not found at {image_path} - text summary only.\n",
                      file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Output was piped into something that closed early (e.g. `head`).
        sys.exit(0)
