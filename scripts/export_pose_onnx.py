#!/usr/bin/env python3

"""Export the bundled YOLO pose model to a static FP16 ONNX model."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "src"
    / "emo_robot_perception"
    / "models"
    / "yolo26n-pose.pt"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "src"
    / "emo_robot_perception"
    / "models"
    / "yolo26n-pose-fp16.onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export yolo26n-pose.pt as a batch-1 static FP16 ONNX model "
            "and verify it with ONNX Runtime CUDA."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="source Ultralytics .pt model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination .onnx model",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="static square model input size (default: 640)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()

    if source.suffix.lower() != ".pt" or not source.is_file():
        raise SystemExit(f"Input .pt model does not exist: {source}")
    if destination.suffix.lower() != ".onnx":
        raise SystemExit(f"Output path must end in .onnx: {destination}")
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be positive")

    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable in the active Python environment; "
            "FP16 GPU export was not attempted."
        )
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise SystemExit(
            "ONNX Runtime CUDAExecutionProvider is unavailable. Run "
            "scripts/install_onnxruntime_jetson.sh first."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(source))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            quantize=16,
            simplify=False,
            opset=20,
            device=0,
        )
    ).resolve()
    if exported_path != destination:
        shutil.move(str(exported_path), str(destination))

    model_onnx = onnx.load(str(destination))
    onnx.checker.check_model(model_onnx)
    model_input = model_onnx.graph.input[0]
    input_shape = [
        dimension.dim_value
        for dimension in model_input.type.tensor_type.shape.dim
    ]
    expected_shape = [1, 3, args.imgsz, args.imgsz]
    if input_shape != expected_shape:
        raise SystemExit(
            f"Unexpected ONNX input shape: {input_shape}; "
            f"expected {expected_shape}"
        )
    if model_input.type.tensor_type.elem_type != onnx.TensorProto.FLOAT16:
        raise SystemExit("Exported ONNX model input is not FP16.")

    session = ort.InferenceSession(
        str(destination),
        providers=[
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ],
    )
    active_providers = session.get_providers()
    if not active_providers or active_providers[0] != "CUDAExecutionProvider":
        raise SystemExit(
            "ONNX session did not select CUDAExecutionProvider: "
            f"{active_providers}"
        )
    input_name = session.get_inputs()[0].name
    outputs = session.run(
        None,
        {input_name: np.zeros(expected_shape, dtype=np.float16)},
    )
    if not outputs:
        raise SystemExit("ONNX Runtime smoke test returned no outputs.")

    size_mib = destination.stat().st_size / (1024 * 1024)
    print(f"ONNX model: {destination}")
    print(f"Size: {size_mib:.1f} MiB")
    print(f"Input: {input_name} {expected_shape} float16")
    print(f"Providers: {active_providers}")
    print("CUDA inference smoke test passed.")


if __name__ == "__main__":
    main()
