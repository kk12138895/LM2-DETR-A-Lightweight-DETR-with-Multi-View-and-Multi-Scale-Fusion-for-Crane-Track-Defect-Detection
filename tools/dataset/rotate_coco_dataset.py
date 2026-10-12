#!/usr/bin/env python3
"""Create a rotated COCO dataset without modifying the source directory."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - exercised before real work starts
    raise SystemExit(
        "Pillow is required. Install it with `pip install pillow`, or run this "
        "script with an environment that already has PIL/Pillow."
    ) from exc


DEFAULT_SRC = "/mnt/e/project/D-FINE-master/datasets/rail-mix-coco"
DEFAULT_DST = "/mnt/e/project/D-FINE-master/datasets/rail-mix-coco-rotated"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_path(path_text: str) -> Path:
    """Resolve normal paths, plus WSL-style /mnt/<drive>/... paths on Windows."""
    path = Path(path_text).expanduser()
    if path.exists() or not path_text.startswith("/mnt/"):
        return path

    parts = Path(path_text).parts
    if len(parts) >= 3 and len(parts[2]) == 1:
        drive = parts[2].upper()
        return Path(f"{drive}:/" + "/".join(parts[3:]))
    return path


def parse_angles(values: list[int]) -> list[int]:
    angles = []
    for value in values:
        angle = value % 360
        if angle not in {0, 90, 180, 270}:
            raise argparse.ArgumentTypeError("Only 0, 90, 180, and 270 degrees are supported.")
        if angle not in angles:
            angles.append(angle)
    return angles


def rotated_size(width: float, height: float, angle: int) -> tuple[float, float]:
    if angle in {90, 270}:
        return height, width
    return width, height


def rotate_point(x: float, y: float, width: float, height: float, angle: int) -> tuple[float, float]:
    if angle == 0:
        return x, y
    if angle == 90:
        return y, width - x
    if angle == 180:
        return width - x, height - y
    if angle == 270:
        return height - y, x
    raise ValueError(f"Unsupported angle: {angle}")


def rotate_bbox(bbox: list[float], width: float, height: float, angle: int) -> list[float]:
    x, y, w, h = bbox
    corners = [
        rotate_point(x, y, width, height, angle),
        rotate_point(x + w, y, width, height, angle),
        rotate_point(x, y + h, width, height, angle),
        rotate_point(x + w, y + h, width, height, angle),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    new_width, new_height = rotated_size(width, height, angle)
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(new_width, max(xs))
    y2 = min(new_height, max(ys))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def output_name(file_name: str, angle: int) -> str:
    source = Path(file_name)
    if angle == 0:
        return file_name
    suffix = source.suffix if source.suffix.lower() in IMAGE_EXTS else ".jpg"
    return f"{source.stem}_rot{angle}{suffix}"


def rotate_image(source: Path, destination: Path, angle: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if angle == 0:
        shutil.copy2(source, destination)
        return

    with Image.open(source) as image:
        if angle == 90:
            rotated = image.transpose(Image.Transpose.ROTATE_90)
        elif angle == 180:
            rotated = image.transpose(Image.Transpose.ROTATE_180)
        elif angle == 270:
            rotated = image.transpose(Image.Transpose.ROTATE_270)
        else:
            raise ValueError(f"Unsupported angle: {angle}")
        rotated.save(destination)


def build_split(
    source_root: Path,
    output_root: Path,
    split: str,
    angles: list[int],
    include_original: bool,
) -> tuple[int, int]:
    ann_path = source_root / "Annotations" / f"{split}.json"
    if not ann_path.exists():
        print(f"[skip] {split}: annotation file not found: {ann_path}")
        return 0, 0

    with ann_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    output_angles = ([0] if include_original else []) + angles
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        annotations_by_image[ann["image_id"]].append(ann)

    new_coco = {key: copy.deepcopy(value) for key, value in coco.items() if key not in {"images", "annotations"}}
    new_coco["images"] = []
    new_coco["annotations"] = []

    next_image_id = 1
    next_ann_id = 1

    for image_info in coco.get("images", []):
        source_image = source_root / split / image_info["file_name"]
        if not source_image.exists():
            print(f"[warn] Missing image, skipped: {source_image}")
            continue

        width = float(image_info["width"])
        height = float(image_info["height"])

        for angle in output_angles:
            new_file_name = output_name(image_info["file_name"], angle)
            destination_image = output_root / split / new_file_name
            rotate_image(source_image, destination_image, angle)

            new_width, new_height = rotated_size(width, height, angle)
            new_image = copy.deepcopy(image_info)
            new_image["id"] = next_image_id
            new_image["file_name"] = new_file_name
            new_image["width"] = int(new_width)
            new_image["height"] = int(new_height)
            new_coco["images"].append(new_image)

            for ann in annotations_by_image.get(image_info["id"], []):
                new_ann = copy.deepcopy(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_image_id
                new_ann["bbox"] = rotate_bbox(ann["bbox"], width, height, angle)
                new_ann["area"] = new_ann["bbox"][2] * new_ann["bbox"][3]
                if "segmentation" in new_ann and new_ann["segmentation"]:
                    raise SystemExit(
                        "This script updates bbox annotations only, but segmentation data was found. "
                        "Remove segmentation fields or extend the script before using it on masks."
                    )
                new_coco["annotations"].append(new_ann)
                next_ann_id += 1

            next_image_id += 1

    output_ann_dir = output_root / "Annotations"
    output_ann_dir.mkdir(parents=True, exist_ok=True)
    with (output_ann_dir / f"{split}.json").open("w", encoding="utf-8") as handle:
        json.dump(new_coco, handle, ensure_ascii=False, indent=2)

    return len(new_coco["images"]), len(new_coco["annotations"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a rotated COCO dataset in a new directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help="Source COCO dataset root.")
    parser.add_argument("--dst", default=DEFAULT_DST, help="Output COCO dataset root.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to process.")
    parser.add_argument("--angles", nargs="+", type=int, default=[90, 180, 270], help="Rotation angles.")
    parser.add_argument(
        "--no-include-original",
        action="store_true",
        help="Do not copy original images into the output dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory first if it exists.")
    args = parser.parse_args()

    source_root = resolve_path(args.src)
    output_root = resolve_path(args.dst)
    angles = parse_angles(args.angles)

    if not source_root.exists():
        raise SystemExit(f"Source dataset does not exist: {source_root}")
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {output_root}\nUse --overwrite to replace it.")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=False)
    include_original = not args.no_include_original

    for split in args.splits:
        images, annotations = build_split(source_root, output_root, split, angles, include_original)
        if images:
            print(f"[ok] {split}: wrote {images} images and {annotations} annotations")

    print(f"Done. Output dataset: {output_root}")


if __name__ == "__main__":
    main()
