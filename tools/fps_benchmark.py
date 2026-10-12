"""
D-FINE FPS benchmark (GPU forward-only).
Usage:
  python tools/fps_benchmark.py -c configs/dfine/custom/dfine_hgnetv2_n_ghost_c3k2_fps.yml -r path/to.pth
"""
import argparse
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# allow running in envs without dataset deps (e.g. yolo01 + CUDA)
if 'faster_coco_eval' not in sys.modules:
    _fce = MagicMock()
    _fce.core.mask = MagicMock()
    sys.modules['faster_coco_eval'] = _fce
    sys.modules['faster_coco_eval.core'] = _fce.core
    sys.modules['faster_coco_eval.core.mask'] = _fce.core.mask
if 'calflops' not in sys.modules:
    sys.modules['calflops'] = MagicMock()

from src.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--deploy', action='store_true', help='call model.deploy() before timing')
    parser.add_argument('-u', '--update', nargs='+', help='override yaml, e.g. num_classes=5')
    args = parser.parse_args()

    extra = {}
    if args.update:
        from src.core.yaml_utils import parse_cli
        extra = parse_cli(args.update)
    cfg = YAMLConfig(args.config, resume=args.resume, use_ema=False, **extra)
    model = cfg.model
    if 'HGNetv2' in cfg.yaml_cfg:
        print('HGNetv2.block_conv =', cfg.yaml_cfg['HGNetv2'].get('block_conv', 'conv'))
    if 'DFINE' in cfg.yaml_cfg:
        print('DFINE.encoder =', cfg.yaml_cfg['DFINE'].get('encoder', 'HybridEncoder'))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    if args.deploy and hasattr(model, 'deploy'):
        model.deploy()

    ckpt = torch.load(args.resume, map_location='cpu')
    state = ckpt.get('ema', {}).get('module') or ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'load: missing={len(missing)}, unexpected={len(unexpected)}')
    if missing:
        print('  missing sample:', missing[:5])

    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    elapsed = time.time() - t0
    ms = elapsed / args.iters * 1000
    print(f'device={device}, imgsz={args.imgsz}, iters={args.iters}')
    print(f'inference: {ms:.2f} ms/im, FPS: {1000/ms:.2f}')


if __name__ == '__main__':
    main()
