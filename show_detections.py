"""
show_detections.py

Visualize MegaDetector bounding boxes on top of camera trap images.

Expects a detections CSV with columns:
    MetaId, Name, CameraPath, category, conf, img_w, img_h,
    bbox_x, bbox_y, bbox_w, bbox_h, multiple

Bounding boxes are stored as ratios (0-1) relative to image width/height,
with (bbox_x, bbox_y) as the top-left corner.

Usage as a script:
    python show_detections.py detections_out.csv --images-root jasper-wildlife/ct_photos

Usage as a library:
    from show_detections import show_image, show_random_images

    show_image("CAMERA_ARRAY_G/camera_G1/p_026288.jpg20220702.jpg",
               csv_path="detections_out.csv",
               images_root="jasper-wildlife/ct_photos")

    show_random_images(4, csv_path="detections_out.csv",
                        images_root="jasper-wildlife/ct_photos")
"""

import argparse
import os
import random

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Category colors so different classes are visually distinct
CATEGORY_COLORS = {
    "animal": "#00E676",   # green
    "person": "#FF5252",   # red
    "vehicle": "#40C4FF",  # blue
}
DEFAULT_COLOR = "#FFD740"  # amber, for any other/unknown category


def _resolve_path(row, images_root):
    """Build the full image path from a detections dataframe row."""
    return os.path.join(images_root, row["CameraPath"], row["Name"])


def _draw_boxes_on_axis(ax, image, detections):
    """Draw all detection boxes + labels for one image onto a matplotlib axis."""
    ax.imshow(image)
    img_w, img_h = image.size

    for _, det in detections.iterrows():
        # Convert ratio-based bbox to pixel coordinates
        x = det["bbox_x"] * img_w
        y = det["bbox_y"] * img_h
        w = det["bbox_w"] * img_w
        h = det["bbox_h"] * img_h

        color = CATEGORY_COLORS.get(det["category"], DEFAULT_COLOR)

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        label = f'{det["category"]} {det["conf"]:.2f}'
        ax.text(
            x, max(y - 4, 0), label,
            color="black",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor=color, edgecolor="none", pad=1.5, alpha=0.9),
            verticalalignment="bottom",
        )

    ax.set_xticks([])
    ax.set_yticks([])


def show_image(image_name, csv_path="detections_out.csv", images_root=".", ax=None):
    """
    Display a single image with all of its bounding boxes drawn on top.

    image_name: the value in the CSV's "Name" column (e.g. "p_026288.jpg20220702.jpg")
                If multiple rows share this name (e.g. from different CameraPaths),
                all matching rows are used.
    csv_path:   path to the detections CSV
    images_root: root folder that CameraPath is relative to
    ax:         optional matplotlib axis to draw on (used internally by show_random_images).
                If None, opens its own figure and calls plt.show().
    """
    df = pd.read_csv(csv_path)
    detections = df[df["Name"] == image_name]

    if detections.empty:
        raise ValueError(f"No rows found in {csv_path} for image '{image_name}'")

    row = detections.iloc[0]
    image_path = _resolve_path(row, images_root)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found on disk: {image_path}")

    image = Image.open(image_path).convert("RGB")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 8))

    _draw_boxes_on_axis(ax, image, detections)
    ax.set_title(image_name, fontsize=10)

    if standalone:
        plt.tight_layout()
        plt.show()


def show_random_images(n, csv_path="detections_out.csv", images_root=".", seed=None):
    """
    Display n random images (each with all of their bounding boxes) in a grid.

    n:          number of unique images to show. If n is greater than the number
                of unique images available, all available images are shown.
    csv_path:   path to the detections CSV
    images_root: root folder that CameraPath is relative to
    seed:       optional random seed for reproducibility
    """
    df = pd.read_csv(csv_path)
    unique_names = df["Name"].unique().tolist()

    if seed is not None:
        random.seed(seed)

    n = min(n, len(unique_names))
    chosen = random.sample(unique_names, n)

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

    # Normalize axes to a flat list regardless of grid shape
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else list(axes)

    for ax, name in zip(axes, chosen):
        try:
            show_image(name, csv_path=csv_path, images_root=images_root, ax=ax)
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center", wrap=True, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide any unused subplot axes
    for ax in axes[len(chosen):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MegaDetector bounding boxes.")
    parser.add_argument("csv_path", help="Path to detections CSV")
    parser.add_argument("--images-root", default=".", help="Root folder for CameraPath")
    parser.add_argument("--n", type=int, default=4, help="Number of random images to show")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    show_random_images(args.n, csv_path=args.csv_path, images_root=args.images_root, seed=args.seed)