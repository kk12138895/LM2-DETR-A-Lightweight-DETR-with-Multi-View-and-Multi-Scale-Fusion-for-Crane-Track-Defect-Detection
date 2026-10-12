"""
Batch inference visualization with prediction coordinates.

This script saves:
  1. annotated images with class, score, and xyxy coordinates;
  2. predictions.csv;
  3. predictions.json.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.core import YAMLConfig


DEFAULT_CONFIG = "/mnt/e/project/D-FINE-master/configs/dfine/custom/dfine_hgnetv2_n_custom_new.yml"
DEFAULT_RESUME = "/mnt/e/project/D-FINE-master/output/lightcwam2/best_stg2.pth"
DEFAULT_INPUT = "/mnt/e/project/D-FINE-master/datasets/rail-mix-coco/val"
DEFAULT_ANN = "/mnt/e/project/D-FINE-master/datasets/rail-mix-coco/Annotations/val.json"
DEFAULT_OUTPUT = "/mnt/e/project/D-FINE-master/output/lightcwam2/val_visualized_with_coords"


def resolve_path(path):
    """Keep Linux paths on Linux, map /mnt/e/... to E:/... when running on Windows."""
    if path is None:
        return None
    path = str(path)
    if os.name == "nt" and path.startswith("/mnt/") and len(path) > 6:
        drive = path[5]
        rest = path[7:].replace("/", "\\")
        return f"{drive.upper()}:\\{rest}"
    return path


def load_class_names(ann_file):
    if not ann_file or not os.path.exists(ann_file):
        return {}
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    categories = sorted(data.get("categories", []), key=lambda item: item["id"])
    return {idx: cat.get("name", str(cat["id"])) for idx, cat in enumerate(categories)}


def get_images(input_path):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in suffixes)


def load_font(size=12):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_predictions(image, detections, color=(255, 0, 0)):
    draw = ImageDraw.Draw(image)
    font = load_font(size=max(10, image.width // 35))

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        text = (
            f"{det['label_name']} {det['score']:.2f} "
            f"({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
        )

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - text_h - 3)
        draw.rectangle([x1, text_y, x1 + text_w + 4, text_y + text_h + 3], fill=(255, 255, 255))
        draw.text((x1 + 2, text_y), text, fill=(0, 0, 255), font=font)

    return image


def build_model(config, resume, device):
    cfg = YAMLConfig(config, resume=resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(resume, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    return Model().to(device).eval()


def run(args):
    config = resolve_path(args.config)
    resume = resolve_path(args.resume)
    input_path = resolve_path(args.input)
    ann_file = resolve_path(args.ann_file)
    output_dir = Path(resolve_path(args.output_dir))
    image_out_dir = output_dir / "images"
    image_out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    class_names = load_class_names(ann_file)
    model = build_model(config, resume, device)
    transforms = T.Compose([T.Resize((args.input_size, args.input_size)), T.ToTensor()])

    images = get_images(input_path)
    if args.max_images > 0:
        images = images[: args.max_images]

    all_predictions = []
    csv_rows = []

    with torch.no_grad():
        for idx, image_path in enumerate(images, start=1):
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            orig_size = torch.tensor([[width, height]], device=device)
            image_tensor = transforms(image).unsqueeze(0).to(device)

            labels, boxes, scores = model(image_tensor, orig_size)
            labels = labels[0].detach().cpu()
            boxes = boxes[0].detach().cpu()
            scores = scores[0].detach().cpu()

            keep = scores >= args.threshold
            labels = labels[keep]
            boxes = boxes[keep]
            scores = scores[keep]

            detections = []
            for det_idx, (label, box, score) in enumerate(zip(labels, boxes, scores), start=1):
                label_id = int(label.item())
                label_name = class_names.get(label_id, str(label_id))
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                x1 = max(0.0, min(x1, width))
                y1 = max(0.0, min(y1, height))
                x2 = max(0.0, min(x2, width))
                y2 = max(0.0, min(y2, height))
                score_value = float(score.item())

                det = {
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "det_id": det_idx,
                    "label_id": label_id,
                    "label_name": label_name,
                    "score": score_value,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                    "image_width": width,
                    "image_height": height,
                }
                detections.append(det)
                csv_rows.append(
                    {
                        "image": image_path.name,
                        "det_id": det_idx,
                        "label_id": label_id,
                        "label_name": label_name,
                        "score": f"{score_value:.6f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                        "w": f"{x2 - x1:.2f}",
                        "h": f"{y2 - y1:.2f}",
                        "image_width": width,
                        "image_height": height,
                    }
                )

            draw_predictions(image, detections)
            out_image_path = image_out_dir / image_path.name
            image.save(out_image_path)

            all_predictions.append(
                {
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "output_image": str(out_image_path),
                    "detections": detections,
                }
            )

            if idx % args.print_freq == 0 or idx == len(images):
                print(f"Processed {idx}/{len(images)} images")

    json_path = output_dir / "predictions.json"
    csv_path = output_dir / "predictions.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "image",
        "det_id",
        "label_id",
        "label_name",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "w",
        "h",
        "image_width",
        "image_height",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Annotated images saved to: {image_out_dir}")
    print(f"Coordinates CSV saved to: {csv_path}")
    print(f"Coordinates JSON saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    parser.add_argument("-r", "--resume", default=DEFAULT_RESUME)
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT)
    parser.add_argument("--ann-file", default=DEFAULT_ANN)
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("-d", "--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument("--print-freq", type=int, default=20)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
