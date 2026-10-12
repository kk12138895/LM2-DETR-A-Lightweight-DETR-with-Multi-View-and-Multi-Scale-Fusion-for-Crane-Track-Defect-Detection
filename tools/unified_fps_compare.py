"""Unified forward FPS: same protocol for YOLO (ultralytics) and D-FINE."""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import torch

DFINE_ROOT = Path(__file__).resolve().parents[1]
ULTRA_ROOT = DFINE_ROOT.parent / "ultralytics-main"
FPS_DIR = DFINE_ROOT.parent / "训练结果" / "fps测试2"

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

from src.core import YAMLConfig

WARMUP = 20
ITERS = 200
IMGSZ = 640
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DFINE_CASES = [
    ("D-FINE-N", "configs/dfine/custom/dfine_hgnetv2_n_custom.yml", "DFINEn_best_stg2.pth"),
    ("D-FINE-S", "configs/dfine/custom/dfine_hgnetv2_s_fps.yml", "DFINEs_best_stg1.pth"),
    ("D-FINE-N-GhostC3K2", "configs/dfine/custom/dfine_hgnetv2_n_ghost_c3k2_fps.yml",
     "dfine_hgnetv2_n_GhostConv123_Native_C3K2 0.5 2.pth"),
]

YOLO_CASES = [
    ("YOLOv5", "yolov5_best.pt"),
    ("YOLOv6", "yolov6_best.pt"),
    ("YOLOv7-tiny", "yolov7_tiny_best.pt"),
    ("YOLOv10", "yolov10best.pt"),
    ("YOLOv11", "yolov11best.pt"),
]


def measure(run_fn, device):
    dev = torch.device(device)
    x = torch.randn(1, 3, IMGSZ, IMGSZ, device=dev)
    with torch.no_grad():
        for _ in range(WARMUP):
            run_fn(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(ITERS):
            run_fn(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    ms = (time.time() - t0) / ITERS * 1000
    return ms, 1000 / ms


def bench_yolo(name, weight_name):
    from ultralytics import YOLO

    w = FPS_DIR / weight_name
    if not w.exists():
        return None, f"missing {w}"
    try:
        yolo = YOLO(str(w), task="detect")
        net = yolo.model.to(DEVICE).eval()
        if hasattr(net, "fuse"):
            net.fuse()

        def run(x):
            net(x)

        ms, fps = measure(run, DEVICE)
        return (name, ms, fps, "ok"), None
    except Exception as e:
        return None, str(e)


def bench_dfine(name, cfg_rel, weight_name):
    w = FPS_DIR / weight_name
    if not w.exists():
        return None, f"missing {w}"
    try:
        cfg = YAMLConfig(str(DFINE_ROOT / cfg_rel), resume=str(w), use_ema=False)
        model = cfg.model.to(DEVICE).eval()
        ckpt = torch.load(w, map_location="cpu", weights_only=False)
        state = ckpt.get("ema", {}).get("module") or ckpt.get("model", ckpt)
        missing, _ = model.load_state_dict(state, strict=False)
        status = "ok" if len(missing) == 0 else f"ok(missing={len(missing)})"

        def run(x):
            model(x)

        ms, fps = measure(run, DEVICE)
        return (name, ms, fps, status), None
    except Exception as e:
        return None, str(e)


def main():
    print(f"Protocol: forward-only | device={DEVICE} | imgsz={IMGSZ} | warmup={WARMUP} | iters={ITERS}")
    print(f"weights_dir={FPS_DIR}\n")
    rows = []

    for name, cfg, wt in DFINE_CASES:
        r, err = bench_dfine(name, cfg, wt)
        if r:
            rows.append(("D-FINE",) + r)
        else:
            rows.append(("D-FINE", name, None, None, f"FAIL: {err}"))

    for name, wt in YOLO_CASES:
        r, err = bench_yolo(name, wt)
        if r:
            rows.append(("YOLO",) + r)
        else:
            rows.append(("YOLO", name, None, None, f"FAIL: {err}"))

    print(f"{'Family':<10} {'Model':<22} {'ms/im':>10} {'FPS':>10} {'Note':>12}")
    print("-" * 68)
    for fam, name, ms, fps, note in rows:
        if ms is None:
            print(f"{fam:<10} {name:<22} {'—':>10} {'—':>10} {note}")
        else:
            print(f"{fam:<10} {name:<22} {ms:>10.2f} {fps:>10.2f} {note}")

    yolo_ok = [(n, ms, fps) for fam, n, ms, fps, note in rows if fam == "YOLO" and ms is not None]
    dfine_ok = [(n, ms, fps) for fam, n, ms, fps, note in rows if fam == "D-FINE" and ms is not None]
    if yolo_ok and dfine_ok:
        y_ms = sum(x[1] for x in yolo_ok) / len(yolo_ok)
        d_ms = sum(x[1] for x in dfine_ok) / len(dfine_ok)
        print(f"\n平均 forward 耗时: YOLO {y_ms:.2f} ms | D-FINE {d_ms:.2f} ms | 差值 {y_ms - d_ms:+.2f} ms (YOLO−D-FINE)")


if __name__ == "__main__":
    main()
