"""
FPS using Ultralytics-style timing: Profile + per-image average (same as val speed['inference']).
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import torch

DFINE_ROOT = Path(__file__).resolve().parents[1]
ULTRA_ROOT = DFINE_ROOT.parent / "ultralytics-main"
FPS_DIR = DFINE_ROOT.parent / "训练结果" / "fps测试2"
COCO8_YAML = DFINE_ROOT.parent / "训练结果" / "fps测试" / "datasets" / "coco8_fps.yaml"

sys.path.insert(0, str(DFINE_ROOT))
sys.path.insert(0, str(ULTRA_ROOT))

if "faster_coco_eval" not in sys.modules:
    _fce = MagicMock()
    _fce.core.mask = MagicMock()
    sys.modules["faster_coco_eval"] = _fce
    sys.modules["faster_coco_eval.core"] = _fce.core
    sys.modules["faster_coco_eval.core.mask"] = _fce.core.mask
if "calflops" not in sys.modules:
    sys.modules["calflops"] = MagicMock()

from ultralytics.utils.ops import Profile

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMGSZ = 640


def ms_fps(ms):
    return ms, 1000.0 / ms if ms > 0 else 0.0


def yolo_ultra_fps(name, weight):
    from ultralytics import YOLO

    w = FPS_DIR / weight
    if not w.exists():
        return None, f"missing {w}"
    try:
        model = YOLO(str(w), task="detect")
        metrics = model.val(
            data=str(COCO8_YAML),
            imgsz=IMGSZ,
            batch=1,
            device=0 if torch.cuda.is_available() else "cpu",
            plots=False,
            verbose=False,
        )
        ms = float(metrics.speed["inference"])
        return (name, ms, 1000.0 / ms, "ultralytics val"), None
    except Exception as e:
        return None, str(e)


def dfine_ultra_fps(name, cfg_rel, weight):
    import cv2
    import numpy as np
    import torchvision.transforms as T
    from PIL import Image
    from src.core import YAMLConfig
    from src.data.transforms import ConvertPILImage

    w = FPS_DIR / weight
    img_dir = COCO8_YAML.parent / "coco8" / "images" / "val"
    if not w.exists():
        return None, f"missing {w}"
    if not img_dir.is_dir():
        return None, f"missing {img_dir}"
    try:
        cfg = YAMLConfig(
            str(DFINE_ROOT / cfg_rel),
            use_ema=False,
            device=DEVICE,
            num_classes=5,
        )
        model = cfg.model.to(DEVICE).eval()
        ckpt = torch.load(w, map_location="cpu", weights_only=False)
        state = ckpt.get("ema", {}).get("module") or ckpt.get("model", ckpt)
        missing, _ = model.load_state_dict(state, strict=False)
        note = "ok" if len(missing) == 0 else f"ok(missing={len(missing)})"

        transform = T.Compose([
            T.Resize((IMGSZ, IMGSZ)),
            ConvertPILImage(dtype="float32", scale=True),
        ])
        paths = sorted(img_dir.glob("*.jpg"))
        with torch.no_grad():
            for p in paths[:1]:
                im0 = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                im = transform(Image.fromarray(im0)).unsqueeze(0).to(DEVICE)
                model(im)
        dt = Profile(device=torch.device(DEVICE))
        with torch.no_grad():
            for p in paths:
                im0 = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                im = transform(Image.fromarray(im0)).unsqueeze(0).to(DEVICE)
                with dt:
                    model(im)
        ms = dt.t / len(paths) * 1000.0
        return (name, ms, 1000.0 / ms, note), None
    except Exception as e:
        return None, str(e)


def main():
    print(f"Ultralytics-style: Profile inference, avg per image | device={DEVICE} | data={COCO8_YAML}")
    print(f"imgsz={IMGSZ}, batch=1 (D-FINE val_dataloader.total_batch_size=1 in fps yml)\n")

    rows = []
    yolo_cases = [
        ("YOLOv5", "yolov5_best.pt"),
        ("YOLOv6", "yolov6_best.pt"),
        ("YOLOv7-tiny", "yolov7_tiny_best.pt"),
        ("YOLOv10", "yolov10best.pt"),
        ("YOLOv11", "yolov11best.pt"),
    ]
    dfine_cases = [
        ("D-FINE-N", "configs/dfine/custom/dfine_hgnetv2_n_ultra_fps.yml", "DFINEn_best_stg2.pth"),
        ("D-FINE-S", "configs/dfine/custom/dfine_hgnetv2_s_fps.yml", "DFINEs_best_stg1.pth"),
        ("D-FINE-N-GhostC3K2", "configs/dfine/custom/dfine_hgnetv2_n_ghost_c3k2_fps.yml",
         "dfine_hgnetv2_n_GhostConv123_Native_C3K2 0.5 2.pth"),
    ]

    for n, w in yolo_cases:
        r, err = yolo_ultra_fps(n, w)
        rows.append(("YOLO", r, err))

    for n, c, w in dfine_cases:
        r, err = dfine_ultra_fps(n, c, w)
        rows.append(("D-FINE", r, err))

    print(f"{'Family':<8} {'Model':<22} {'ms/im':>10} {'FPS':>10} {'Note':>20}")
    print("-" * 74)
    for fam, r, err in rows:
        if r:
            name, ms, fps, note = r
            print(f"{fam:<8} {name:<22} {ms:>10.2f} {fps:>10.2f} {note:>20}")
        else:
            print(f"{fam:<8} {err}")

    yolo_ms = [r[1] for fam, r, e in rows if fam == "YOLO" and r]
    dfine_ms = [r[1] for fam, r, e in rows if fam == "D-FINE" and r]
    if yolo_ms:
        print(f"\nYOLO 平均: {sum(yolo_ms)/len(yolo_ms):.2f} ms, {1000/(sum(yolo_ms)/len(yolo_ms)):.2f} FPS")
    if dfine_ms:
        print(f"D-FINE 平均: {sum(dfine_ms)/len(dfine_ms):.2f} ms, {1000/(sum(dfine_ms)/len(dfine_ms)):.2f} FPS")


if __name__ == "__main__":
    main()
