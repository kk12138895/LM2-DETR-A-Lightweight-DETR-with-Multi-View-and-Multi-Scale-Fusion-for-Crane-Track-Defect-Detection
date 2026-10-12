"""
Analyze parameter counts of major LCR-DETR components.

The report contains two views:
  1. top-level structure: backbone / encoder / decoder / cross-view fusion;
  2. named components used in the ablation table: Ghost, MCSA, C3K2, and
     Cross-view Fusion.
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


DEFAULT_CONFIG = "/mnt/e/project/D-FINE-master/configs/dfine/custom/dfine_hgnetv2_n_custom_new.yml"
DEFAULT_RESUME = "/mnt/e/project/D-FINE-master/output/lightcwam2/best_stg2.pth"
DEFAULT_OUTPUT = "/mnt/e/project/D-FINE-master/output/lightcwam2/module_params.csv"


def resolve_path(path):
    """Map /mnt/e/... to E:/... when this script is run on Windows."""
    if path is None:
        return None
    path = str(path)
    if os.name == "nt" and path.startswith("/mnt/") and len(path) > 6:
        drive = path[5]
        rest = path[7:].replace("/", "\\")
        return f"{drive.upper()}:\\{rest}"
    return path


def unique_params(module):
    seen = set()
    params = []
    for param in module.parameters(recurse=True):
        if id(param) not in seen:
            seen.add(id(param))
            params.append(param)
    return params


def count_params(module):
    return sum(param.numel() for param in unique_params(module))


def count_trainable_params(module):
    return sum(param.numel() for param in unique_params(module) if param.requires_grad)


def count_modules(model, module_types):
    total = 0
    details = []
    seen = set()
    for name, module in model.named_modules():
        if isinstance(module, module_types):
            params = []
            for param in module.parameters(recurse=True):
                if id(param) not in seen:
                    seen.add(id(param))
                    params.append(param)
            num = sum(param.numel() for param in params)
            total += num
            details.append((name, module.__class__.__name__, num))
    return total, details


def add_row(rows, group, params, total, trainable=None, note=""):
    if trainable is None:
        trainable = params
    rows.append(
        OrderedDict(
            Group=group,
            Params=params,
            Params_M=f"{params / 1e6:.4f}",
            Ratio=f"{params / total * 100:.2f}%",
            Trainable=trainable,
            Note=note,
        )
    )


def load_model(config, resume, deploy):
    import torch
    from src.core import YAMLConfig

    cfg = YAMLConfig(config, resume=resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(resume, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]
    missing, unexpected = cfg.model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model = cfg.model
    if deploy:
        model = model.deploy()
    return model


def write_csv(path, rows):
    if not path:
        return
    path = resolve_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to: {path}")


def print_table(rows):
    headers = ["Group", "Params", "Params_M", "Ratio", "Note"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    line = " | ".join(header.ljust(widths[header]) for header in headers)
    sep = "-+-".join("-" * widths[header] for header in headers)
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    parser.add_argument("-r", "--resume", default=DEFAULT_RESUME)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Count training-time parameters instead of deployed model parameters.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-module details for Ghost, MCSA, and C3K2.",
    )
    args = parser.parse_args()

    config = resolve_path(args.config)
    resume = resolve_path(args.resume)
    model = load_model(config, resume, deploy=not args.no_deploy)

    from src.nn.backbone.hgnetv2 import GhostConvBNAct
    from src.zoo.dfine.hybrid_encoder_C3K2 import C3K2, NativeSparseAttention, TransformerEncoderLayer

    total = count_params(model)
    rows = []
    add_row(rows, "Total", total, total, count_trainable_params(model), "whole LCR-DETR")

    for name in ["backbone", "encoder", "decoder", "rot180_fusion"]:
        if hasattr(model, name):
            module = getattr(model, name)
            add_row(rows, name, count_params(module), total, count_trainable_params(module), "top-level module")

    ghost_params, ghost_details = count_modules(model, (GhostConvBNAct,))
    mcsa_params, mcsa_details = count_modules(model, (NativeSparseAttention,))
    encoder_layer_params, encoder_layer_details = count_modules(model, (TransformerEncoderLayer,))
    c3k2_params, c3k2_details = count_modules(model, (C3K2,))
    cross_view_params = count_params(model.rot180_fusion) if hasattr(model, "rot180_fusion") else 0

    add_row(rows, "Ghost", ghost_params, total, note="GhostConvBNAct modules in backbone")
    add_row(rows, "MCSA/NSA", mcsa_params, total, note="NativeSparseAttention only")
    add_row(rows, "MCSA block", encoder_layer_params, total, note="attention + FFN + norms")
    add_row(rows, "C3K2", c3k2_params, total, note="C3K2 modules in encoder neck")
    add_row(rows, "Cross-view Fusion", cross_view_params, total, note="rot180_fusion: rotated view + DCGFM")

    print(f"Config: {config}")
    print(f"Checkpoint: {resume}")
    print(f"Mode: {'training-time' if args.no_deploy else 'deploy'} parameters")
    print()
    print_table(rows)
    write_csv(args.output, rows)

    if args.details:
        detail_groups = [
            ("Ghost details", ghost_details),
            ("MCSA/NSA details", mcsa_details),
            ("MCSA block details", encoder_layer_details),
            ("C3K2 details", c3k2_details),
        ]
        for title, details in detail_groups:
            print(f"\n{title}")
            for name, cls_name, params in details:
                print(f"{name:70s} {cls_name:28s} {params:>10,d} ({params / 1e6:.4f} M)")


if __name__ == "__main__":
    main()
