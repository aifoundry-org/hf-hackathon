#!/usr/bin/env python3

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

_MOSAIC_MODEL_URL = (
    "https://github.com/onnx/models/raw/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx"
)
_EP_REGISTERED = False
_GGML_DEVICES = None


def model_cache_dir() -> Path:
    override = os.environ.get("GGONNX_TEST_MODEL_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "ggonnx-test" / "models"


def download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    with urllib.request.urlopen(url) as response, temp_path.open("wb") as out:
        total_bytes = response.headers.get("Content-Length")
        total = int(total_bytes) if total_bytes is not None else None
        downloaded = 0
        bar_width = 32

        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)

            if total:
                filled = min(bar_width, int(bar_width * downloaded / total))
                bar = "#" * filled + "-" * (bar_width - filled)
                percent = 100.0 * downloaded / total
                message = (
                    f"\rDownloading {destination.name} [{bar}] "
                    f"{percent:5.1f}% ({downloaded}/{total} bytes)"
                )
            else:
                message = f"\rDownloading {destination.name}: {downloaded} bytes"

            print(message, end="", file=sys.stderr, flush=True)

    os.replace(temp_path, destination)
    print(file=sys.stderr, flush=True)


def cached_model_path() -> Path:
    destination = model_cache_dir() / "mosaic-9.onnx"
    if not destination.exists():
        download_with_progress(_MOSAIC_MODEL_URL, destination)
    return destination


def register_ep(ep_library: Path) -> None:
    global _EP_REGISTERED
    if _EP_REGISTERED:
        return
    ort.register_execution_provider_library("GGONNX", str(ep_library))
    _EP_REGISTERED = True


def ggml_devices(ep_library: Path):
    global _GGML_DEVICES
    register_ep(ep_library)
    if _GGML_DEVICES is None:
        _GGML_DEVICES = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == "GGMLExecutionProvider"
        ]
    if not _GGML_DEVICES:
        raise RuntimeError("GGMLExecutionProvider was not discovered after library registration")
    return _GGML_DEVICES


def build_session(model_path: Path, provider: str, ep_library: Path | None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if provider == "cpu":
        so.add_provider("CPUExecutionProvider", {})
    elif provider == "ggml":
        if ep_library is None:
            raise ValueError("--ep-library is required for --provider ggml")
        so.add_provider_for_devices(ggml_devices(ep_library), {})
    else:
        raise ValueError(f"unsupported provider: {provider}")
    return ort.InferenceSession(str(model_path), sess_options=so)


def resolve_input_shape(session: ort.InferenceSession) -> tuple[str, int, int]:
    model_input = session.get_inputs()[0]
    dims = model_input.shape
    if len(dims) != 4:
        raise RuntimeError(f"expected 4D NCHW model input, got {dims}")
    channels = dims[1]
    height = dims[2]
    width = dims[3]
    if channels != 3:
        raise RuntimeError(f"expected 3-channel model input, got {dims}")
    if not isinstance(height, int) or not isinstance(width, int):
        raise RuntimeError(f"expected static spatial input shape, got {dims}")
    return model_input.name, height, width


def load_image_as_nchw(path: Path, height: int, width: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return np.transpose(array, (2, 0, 1))[None, :, :, :]


def save_output_image(output: np.ndarray, path: Path) -> None:
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 3:
        raise RuntimeError(f"expected output shaped [1, 3, H, W], got {output.shape}")
    image = np.transpose(output[0], (1, 2, 0))
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mosaic-9.onnx on an image and save the result.")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("output", type=Path, help="Output image path")
    parser.add_argument(
        "--provider",
        choices=("ggml", "cpu"),
        default="ggml",
        help="Execution provider to use",
    )
    parser.add_argument(
        "--ep-library",
        type=Path,
        default=Path("build/libggonnx_ep.so"),
        help="Path to the GGONNX execution provider shared library",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional explicit path to mosaic-9.onnx; otherwise uses the cached download",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model if args.model is not None else cached_model_path()
    ep_library = args.ep_library.resolve() if args.provider == "ggml" else None
    session = build_session(model_path, args.provider, ep_library)
    input_name, height, width = resolve_input_shape(session)
    input_tensor = load_image_as_nchw(args.input, height, width)
    output_name = session.get_outputs()[0].name
    output_tensor = session.run([output_name], {input_name: input_tensor})[0]
    save_output_image(output_tensor, args.output)
    print(
        f"saved {args.output} using {args.provider} "
        f"(model={model_path}, input={input_name}, output={output_name}, size={width}x{height})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
