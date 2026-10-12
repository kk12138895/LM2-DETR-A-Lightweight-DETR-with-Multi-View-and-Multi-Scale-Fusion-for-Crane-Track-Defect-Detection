import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"Total Params: {total_params:,}")
    print(f"Total Params(M): {total_params / 1e6:.2f} M")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"Trainable Params(M): {trainable_params / 1e6:.2f} M")
    print("=" * 60)


def get_base_size(cfg, fallback=640):
    try:
        return int(cfg.train_dataloader.collate_fn.base_size)
    except Exception:
        return int(fallback)


def count_flops(model, input_size):
    try:
        from calflops import calculate_flops
    except ImportError as exc:
        raise ImportError(
            "GFLOPs calculation requires calflops. Install it in the same Python "
            "environment used for this project, for example: pip install calflops"
        ) from exc

    model = model.deploy().eval()
    input_shape = (1, 3, input_size, input_size)

    flops, macs, _ = calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=False,
        output_precision=4,
        print_detailed=False,
    )

    print("=" * 60)
    print(f"Input Shape: {input_shape}")
    print(f"FLOPs: {flops:,}")
    print(f"GFLOPs: {flops / 1e9:.2f}")
    print(f"MACs: {macs:,}")
    print(f"GMACs: {macs / 1e9:.2f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Square input size for FLOPs/GFLOPs. Defaults to config train_dataloader.collate_fn.base_size or 640.",
    )
    args = parser.parse_args()

    from src.core import YAMLConfig

    cfg = YAMLConfig(args.config)
    model = cfg.model
    input_size = args.input_size or get_base_size(cfg)

    count_params(model)
    try:
        count_flops(model, input_size)
    except ImportError as exc:
        print(exc)


if __name__ == "__main__":
    main()
