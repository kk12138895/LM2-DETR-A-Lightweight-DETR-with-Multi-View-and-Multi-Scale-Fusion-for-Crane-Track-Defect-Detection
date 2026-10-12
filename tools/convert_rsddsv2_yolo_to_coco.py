import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def convert_split(root, split, category_name):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image split directory: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label split directory: {label_dir}")

    images = []
    annotations = []
    ann_id = 1

    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as img:
            width, height = img.size

        images.append({
            "file_name": image_path.name,
            "id": image_id,
            "width": width,
            "height": height,
        })

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path}:{line_no} expected 5 YOLO fields, got {len(parts)}")

            class_id, cx, cy, bw, bh = parts
            class_id = int(class_id)
            if class_id != 0:
                raise ValueError(f"{label_path}:{line_no} expected single class id 0, got {class_id}")

            cx = float(cx) * width
            cy = float(cy) * height
            bw = float(bw) * width
            bh = float(bh) * height
            x = max(0.0, cx - bw / 2.0)
            y = max(0.0, cy - bh / 2.0)
            x2 = min(float(width), cx + bw / 2.0)
            y2 = min(float(height), cy + bh / 2.0)
            bw = max(0.0, x2 - x)
            bh = max(0.0, y2 - y)
            if bw <= 0.0 or bh <= 0.0:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": category_name, "supercategory": "defect"}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/RSDDsv2", help="RSDDsv2 dataset root")
    parser.add_argument("--category-name", default="defect", help="COCO category name for YOLO class 0")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / "Annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        coco = convert_split(root, split, args.category_name)
        out_path = out_dir / f"{split}.json"
        out_path.write_text(json.dumps(coco, ensure_ascii=False, indent=4), encoding="utf-8")
        print(
            f"{split}: images={len(coco['images'])}, "
            f"annotations={len(coco['annotations'])}, output={out_path}"
        )


if __name__ == "__main__":
    main()
