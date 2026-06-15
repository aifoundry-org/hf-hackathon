#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import shape_inference
from test_support import (
    assert_all_nodes_run_on_ggml,
    cached_model_path,
    cpu_session,
    end_profiling_profile,
    ggml_session,
)

_MNIST_MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/classification/mnist/model/mnist-8.onnx"
)
_MOSAIC_MODEL_URL = (
    "https://github.com/onnx/models/raw/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx"
)
_TINY_YOLOV3_MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/object_detection_segmentation/tiny-yolov3/model/"
    "tiny-yolov3-11.onnx"
)
_TINY_YOLOV2_MODEL_URL = (
    "https://github.com/onnx/models/raw/refs/heads/main/"
    "validated/vision/object_detection_segmentation/tiny-yolov2/model/"
    "tinyyolov2-8.onnx"
)
_OPENWAKEWORD_ALEXA_URL = (
    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.onnx"
)
_OPENWAKEWORD_EMBEDDING_URL = (
    "https://github.com/dscripka/openWakeWord/releases/download/"
    "v0.5.1/embedding_model.onnx"
)
_ARCFACE_MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/"
    "arcfaceresnet100-8.onnx?download=true"
)
_RESNET18_MODEL_URL = (
    "https://github.com/onnx/models/raw/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/classification/resnet/model/resnet18-v2-7.onnx"
)
_MOBILENETV2_MODEL_URL = (
    "https://github.com/onnx/models/raw/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
)
_SUBPIXEL_SR_MODEL_URL = (
    "https://github.com/onnx/models/raw/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/super_resolution/sub_pixel_cnn_2016/model/"
    "super-resolution-10.onnx"
)
_ULTRAFACE_MODEL_URL = (
    "https://github.com/onnx/models/raw/refs/heads/main/"
    "validated/vision/body_analysis/ultraface/models/version-RFB-640.onnx"
)
_EMOTION_FERPLUS_MODEL_URL = (
    "https://github.com/onnx/models/raw/refs/heads/main/"
    "validated/vision/body_analysis/emotion_ferplus/model/"
    "emotion-ferplus-8.onnx"
)
_SHUFFLENETV2_MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/shufflenet-v2-10/resolve/main/"
    "shufflenet-v2-10.onnx?download=true"
)
_TINYSTORIES_LSTM_MODEL_URL = (
    "https://huggingface.co/phmd/TinyStories-LSTM-5.5M/resolve/main/"
    "tinystories_lstm.onnx?download=true"
)
_TINYSTORIES_LSTM_VOCAB_URL = (
    "https://huggingface.co/phmd/TinyStories-LSTM-5.5M/raw/main/vocab.txt"
)
_WAIFU2X_CUNET_MODEL_URL = (
    "https://huggingface.co/deepghs/waifu2x_onnx/resolve/main/"
    "20250502/onnx_models/cunet/art/noise0_scale2x.onnx?download=true"
)
_WAIFU2X_SWIN_UNET_MODEL_URL = (
    "https://huggingface.co/deepghs/waifu2x_onnx/resolve/main/"
    "20250502/onnx_models/swin_unet/art/noise0_scale2x.onnx?download=true"
)
_SILERO_VAD_MODEL_URL = (
    "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/"
    "model.onnx?download=true"
)


def _esperanto_flag(height: int, width: int) -> np.ndarray:
    # Renders a esperanto flag for testing purposes
    green = np.array([0.0, 154.0, 73.0], dtype=np.float32) / 255.0
    image = np.broadcast_to(green[:, None, None], (3, height, width)).copy()

    canton = min(height, width) // 2
    image[:, :canton, :canton] = 1.0

    cy = cx = canton / 2.0
    r_outer = canton * 0.45
    r_inner = r_outer * np.sin(np.pi / 10.0) / np.sin(7.0 * np.pi / 10.0)
    verts = []
    for k in range(10):
        angle = -np.pi / 2.0 + k * np.pi / 5.0
        r = r_outer if k % 2 == 0 else r_inner
        verts.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))

    ys, xs = np.mgrid[:canton, :canton].astype(np.float32) + 0.5
    inside = np.zeros((canton, canton), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(verts)):
            xi, yi = verts[i]
            xj, yj = verts[i - 1]
            crosses = ((yi > ys) != (yj > ys)) & (
                xs < (xj - xi) * (ys - yi) / (yj - yi) + xi
            )
            inside ^= crosses

    image[:, :canton, :canton][:, inside] = green[:, None]
    return image[None, ...]


def _concretize_input_dims(model_path: Path, overrides: dict[str, list[int]]) -> Path:
    # tiny-yolov3's inputs have dynamic batch and spatial dims (N, 3, ?, ?).
    # GGONNX relies on complete shape information propagated across partition
    # boundaries, so we pin the inputs and re-run shape inference before the
    # model reaches ORT.
    static_path = model_path.with_name(model_path.stem + "_static.onnx")
    if static_path.exists():
        return static_path
    model = onnx.load(str(model_path))
    for graph_input in model.graph.input:
        dims = overrides.get(graph_input.name)
        if dims is None:
            continue
        shape = graph_input.type.tensor_type.shape
        assert len(shape.dim) == len(dims), (
            f"override for {graph_input.name} expects {len(shape.dim)} dims"
        )
        for slot, value in zip(shape.dim, dims):
            slot.ClearField("dim_param")
            slot.dim_value = int(value)
    onnx.save(shape_inference.infer_shapes(model), str(static_path))
    return static_path


def _mnist_one_input() -> np.ndarray:
    image = np.zeros((1, 1, 28, 28), dtype=np.float32)
    image[0, 0, 4:24, 14:16] = 1.0
    return image


def _silero_vad_chunks() -> list[np.ndarray]:
    t = np.arange(512, dtype=np.float32) / 16000.0
    rng = np.random.default_rng(7)
    return [
        np.zeros((1, 512), dtype=np.float32),
        (0.20 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)[None, :],
        (0.10 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)[None, :],
        (0.02 * rng.standard_normal((1, 512))).astype(np.float32),
        np.linspace(-0.25, 0.25, 512, dtype=np.float32)[None, :],
    ]


def _static_input_shape(graph_input) -> list[int]:
    return [
        d if isinstance(d, int) and d > 0 else 1
        for d in graph_input.shape
    ]


def _subpixel_sr_cases(shape: list[int]) -> list[np.ndarray]:
    assert len(shape) == 4, f"expected NCHW input for subpixel SR, got {shape}"
    n, c, h, w = shape
    assert n == 1, f"expected batch size 1 for subpixel SR, got {shape}"
    assert c == 1, f"expected single-channel input for subpixel SR, got {shape}"

    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    gradient = (0.15 + 0.70 * (0.65 * xs + 0.35 * ys)).astype(np.float32)

    checker = (
        0.20 + 0.60 * ((np.indices((h, w)).sum(axis=0) % 2).astype(np.float32))
    ).astype(np.float32)

    rng = np.random.default_rng(11)
    textured = np.clip(
        0.10
        + 0.35 * xs
        + 0.25 * ys
        + 0.20 * np.sin(2.0 * np.pi * xs * 3.0)
        + 0.15 * np.cos(2.0 * np.pi * ys * 5.0)
        + 0.03 * rng.standard_normal((h, w), dtype=np.float32),
        0.0,
        1.0,
    ).astype(np.float32)

    return [
        gradient[None, None, :, :],
        checker[None, None, :, :],
        textured[None, None, :, :],
    ]


def _ultraface_cases(shape: list[int]) -> list[np.ndarray]:
    assert len(shape) == 4, f"expected NCHW input for UltraFace, got {shape}"
    n, c, h, w = shape
    assert n == 1, f"expected batch size 1 for UltraFace, got {shape}"
    assert c == 3, f"expected 3-channel input for UltraFace, got {shape}"

    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    base = np.stack(
        [
            0.10 + 0.70 * xs,
            0.15 + 0.65 * ys,
            0.05 + 0.55 * (0.6 * xs + 0.4 * ys),
        ],
        axis=0,
    ).astype(np.float32)

    stripes = np.stack(
        [
            0.25 + 0.25 * np.sin(2.0 * np.pi * xs * 6.0),
            0.30 + 0.20 * np.cos(2.0 * np.pi * ys * 4.0),
            0.20 + 0.25 * np.sin(2.0 * np.pi * (xs + ys) * 3.0),
        ],
        axis=0,
    ).astype(np.float32)

    rng = np.random.default_rng(23)
    textured = np.clip(
        base
        + stripes
        + 0.03 * rng.standard_normal((c, h, w), dtype=np.float32),
        0.0,
        1.0,
    ).astype(np.float32)

    checker = np.indices((h, w)).sum(axis=0) % 2
    checker_rgb = np.stack(
        [
            0.20 + 0.50 * checker,
            0.15 + 0.45 * (1.0 - checker),
            0.10 + 0.40 * checker,
        ],
        axis=0,
    ).astype(np.float32)

    return [
        base[None, :, :, :],
        checker_rgb[None, :, :, :],
        textured[None, :, :, :],
    ]


def _tiny_yolov2_cases(shape: list[int]) -> list[np.ndarray]:
    assert len(shape) == 4, f"expected NCHW input for tiny YOLOv2, got {shape}"
    n, c, h, w = shape
    assert n == 1, f"expected batch size 1 for tiny YOLOv2, got {shape}"
    assert c == 3, f"expected 3-channel input for tiny YOLOv2, got {shape}"

    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    gradient = np.stack(
        [
            0.05 + 0.85 * xs,
            0.10 + 0.75 * ys,
            0.08 + 0.70 * (0.5 * xs + 0.5 * ys),
        ],
        axis=0,
    ).astype(np.float32)

    checker = (np.indices((h, w)).sum(axis=0) % 2).astype(np.float32)
    checker_rgb = np.stack(
        [
            0.15 + 0.60 * checker,
            0.20 + 0.55 * (1.0 - checker),
            0.10 + 0.50 * checker,
        ],
        axis=0,
    ).astype(np.float32)

    rng = np.random.default_rng(31)
    textured = np.clip(
        gradient
        + np.stack(
            [
                0.18 * np.sin(2.0 * np.pi * xs * 5.0),
                0.16 * np.cos(2.0 * np.pi * ys * 4.0),
                0.14 * np.sin(2.0 * np.pi * (xs + ys) * 2.0),
            ],
            axis=0,
        ).astype(np.float32)
        + 0.025 * rng.standard_normal((c, h, w), dtype=np.float32),
        0.0,
        1.0,
    ).astype(np.float32)

    return [
        gradient[None, :, :, :],
        checker_rgb[None, :, :, :],
        textured[None, :, :, :],
    ]


def _emotion_ferplus_cases(shape: list[int]) -> list[np.ndarray]:
    assert len(shape) == 4, f"expected NCHW input for emotion-ferplus, got {shape}"
    n, c, h, w = shape
    assert n == 1, f"expected batch size 1 for emotion-ferplus, got {shape}"
    assert c == 1, f"expected single-channel input for emotion-ferplus, got {shape}"

    ys, xs = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    radial = np.sqrt(xs * xs + ys * ys)

    smooth = np.clip(0.80 - 0.45 * radial + 0.10 * xs - 0.05 * ys, 0.0, 1.0)

    checker = (
        0.25 + 0.50 * ((np.indices((h, w)).sum(axis=0) % 2).astype(np.float32))
    ).astype(np.float32)

    rng = np.random.default_rng(41)
    textured = np.clip(
        0.55
        - 0.30 * radial
        + 0.18 * np.sin(np.pi * xs * 3.0)
        + 0.12 * np.cos(np.pi * ys * 4.0)
        + 0.03 * rng.standard_normal((h, w), dtype=np.float32),
        0.0,
        1.0,
    ).astype(np.float32)

    return [
        smooth[None, None, :, :].astype(np.float32),
        checker[None, None, :, :].astype(np.float32),
        textured[None, None, :, :].astype(np.float32),
    ]


@pytest.mark.integration
def test_mnist_model_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("mnist-8.onnx", _MNIST_MODEL_URL)
    inputs = {"Input3": _mnist_one_input()}

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    cpu_out = cpu.run(["Plus214_Output_0"], inputs)[0]
    ggml_out = ggml.run(["Plus214_Output_0"], inputs)[0]

    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-5, atol=1e-5)
    assert int(np.argmax(ggml_out, axis=1)[0]) == int(np.argmax(cpu_out, axis=1)[0])


@pytest.mark.integration
def test_mosaic_model_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("mosaic-9.onnx", _MOSAIC_MODEL_URL)
    rng = np.random.default_rng(0)
    inputs = {"input1": rng.standard_normal((1, 3, 224, 224)).astype(np.float32)}

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    cpu_out = cpu.run(["output1"], inputs)[0]
    ggml_out = ggml.run(["output1"], inputs)[0]

    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-3, atol=1e-3)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_tiny_yolov3_model_matches_cpu(ep_library: Path) -> None:
    raw_path = cached_model_path("tiny-yolov3-11.onnx", _TINY_YOLOV3_MODEL_URL)
    model_path = _concretize_input_dims(
        raw_path,
        {"input_1": [1, 3, 416, 416], "image_shape": [1, 2]},
    )
    rng = np.random.default_rng(0)
    inputs = {
        "input_1": rng.standard_normal((1, 3, 416, 416)).astype(np.float32),
        "image_shape": np.array([[416, 416]], dtype=np.float32),
    }
    output_names = ["yolonms_layer_1", "yolonms_layer_1:1", "yolonms_layer_1:2"]

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    # Box coords + class scores must match the CPU reference; the int32 indices
    # tensor is an exact match (NMS output).
    np.testing.assert_allclose(ggml_out[0], cpu_out[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ggml_out[1], cpu_out[1], rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(ggml_out[2], cpu_out[2])


@pytest.mark.integration
def test_tiny_yolov2_matches_cpu_and_is_numerically_stable(ep_library: Path) -> None:
    model_path = cached_model_path("tinyyolov2-8.onnx", _TINY_YOLOV2_MODEL_URL)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    input_info = cpu.get_inputs()[0]
    input_name = input_info.name
    output_names = [out.name for out in cpu.get_outputs()]

    worst_abs = 0.0
    worst_rel = 0.0
    for case in _tiny_yolov2_cases(_static_input_shape(input_info)):
        inputs = {input_name: case}
        cpu_out = cpu.run(output_names, inputs)
        ggml_out = ggml.run(output_names, inputs)

        for got, expected in zip(ggml_out, cpu_out):
            diff = np.abs(got - expected)
            worst_abs = max(worst_abs, float(diff.max(initial=0.0)))
            worst_rel = max(
                worst_rel,
                float(
                    (
                        diff
                        / np.maximum(np.abs(expected), np.float32(1e-6))
                    ).max(initial=0.0)
                ),
            )
            np.testing.assert_allclose(
                got,
                expected,
                rtol=2e-3,
                atol=2e-4,
                err_msg=(
                    f"tiny YOLOv2 drift exceeded tolerance; "
                    f"worst_abs={worst_abs:.3e}, worst_rel={worst_rel:.3e}"
                ),
            )

    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_openwakeword_alexa_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("alexa_v0.1.onnx", _OPENWAKEWORD_ALEXA_URL)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    rng = np.random.default_rng(0)
    inputs = {}
    for graph_input in cpu.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in graph_input.shape]
        inputs[graph_input.name] = rng.standard_normal(shape).astype(np.float32)

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_arcface_resnet100_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("arcfaceresnet100-8.onnx", _ARCFACE_MODEL_URL)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    input_info = cpu.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_info.shape]
    rng = np.random.default_rng(0)
    inputs = {input_info.name: rng.standard_normal(shape).astype(np.float32)}

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_resnet18_v2_matches_cpu(ep_library: Path) -> None:
    raw_path = cached_model_path("resnet18-v2-7.onnx", _RESNET18_MODEL_URL)
    model_path = _concretize_input_dims(raw_path, {"data": [1, 3, 224, 224]})

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    rng = np.random.default_rng(0)
    inputs = {"data": rng.standard_normal((1, 3, 224, 224)).astype(np.float32)}

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_mobilenetv2_matches_cpu(ep_library: Path) -> None:
    raw_path = cached_model_path("mobilenetv2-12.onnx", _MOBILENETV2_MODEL_URL)
    model_path = _concretize_input_dims(raw_path, {"input": [1, 3, 224, 224]})

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    rng = np.random.default_rng(0)
    inputs = {"input": rng.standard_normal((1, 3, 224, 224)).astype(np.float32)}

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_subpixel_sr_matches_cpu_and_is_numerically_stable(ep_library: Path) -> None:
    model_path = cached_model_path("super-resolution-10.onnx", _SUBPIXEL_SR_MODEL_URL)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    input_info = cpu.get_inputs()[0]
    input_name = input_info.name
    output_names = [out.name for out in cpu.get_outputs()]

    worst_abs = 0.0
    worst_rel = 0.0
    for case in _subpixel_sr_cases(_static_input_shape(input_info)):
        inputs = {input_name: case}
        cpu_out = cpu.run(output_names, inputs)
        ggml_out = ggml.run(output_names, inputs)

        for got, expected in zip(ggml_out, cpu_out):
            diff = np.abs(got - expected)
            worst_abs = max(worst_abs, float(diff.max(initial=0.0)))
            worst_rel = max(
                worst_rel,
                float(
                    (
                        diff
                        / np.maximum(np.abs(expected), np.float32(1e-6))
                    ).max(initial=0.0)
                ),
            )
            np.testing.assert_allclose(
                got,
                expected,
                rtol=1e-3,
                atol=1e-4,
                err_msg=(
                    f"subpixel SR drift exceeded tolerance; "
                    f"worst_abs={worst_abs:.3e}, worst_rel={worst_rel:.3e}"
                ),
            )

    profile = end_profiling_profile(ggml)
    ggml_events = [
        event
        for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "GGMLExecutionProvider"
    ]
    assert ggml_events, "expected at least one GGML node event in subpixel SR profile"

    cpu_non_shape_events = [
        event
        for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "CPUExecutionProvider"
        and event.get("args", {}).get("op_name") not in {"Reshape", "Transpose"}
    ]
    assert not cpu_non_shape_events, (
        "expected only shape shuffles on CPU in subpixel SR profile, "
        f"found: {cpu_non_shape_events}"
    )


@pytest.mark.integration
def test_ultraface_matches_cpu_and_is_numerically_stable(ep_library: Path) -> None:
    model_path = cached_model_path("version-RFB-640.onnx", _ULTRAFACE_MODEL_URL)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    input_info = cpu.get_inputs()[0]
    input_name = input_info.name
    output_names = [out.name for out in cpu.get_outputs()]

    worst_abs = 0.0
    worst_rel = 0.0
    for case in _ultraface_cases(_static_input_shape(input_info)):
        inputs = {input_name: case}
        cpu_out = cpu.run(output_names, inputs)
        ggml_out = ggml.run(output_names, inputs)

        for got, expected in zip(ggml_out, cpu_out):
            diff = np.abs(got - expected)
            worst_abs = max(worst_abs, float(diff.max(initial=0.0)))
            worst_rel = max(
                worst_rel,
                float(
                    (
                        diff
                        / np.maximum(np.abs(expected), np.float32(1e-6))
                    ).max(initial=0.0)
                ),
            )
            np.testing.assert_allclose(
                got,
                expected,
                rtol=2e-3,
                atol=2e-4,
                err_msg=(
                    f"UltraFace drift exceeded tolerance; "
                    f"worst_abs={worst_abs:.3e}, worst_rel={worst_rel:.3e}"
                ),
            )

    profile = end_profiling_profile(ggml)
    ggml_events = [
        event
        for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "GGMLExecutionProvider"
    ]
    assert ggml_events, "expected at least one GGML node event in UltraFace profile"

    cpu_conv_events = [
        event
        for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "CPUExecutionProvider"
        and event.get("args", {}).get("op_name") == "Conv"
    ]
    assert not cpu_conv_events, (
        f"found CPU Conv node events in UltraFace profile: {cpu_conv_events}"
    )


@pytest.mark.integration
def test_emotion_ferplus_matches_cpu_and_is_numerically_stable(
    ep_library: Path,
) -> None:
    model_path = cached_model_path("emotion-ferplus-8.onnx", _EMOTION_FERPLUS_MODEL_URL)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    input_info = cpu.get_inputs()[0]
    input_name = input_info.name
    output_names = [out.name for out in cpu.get_outputs()]

    worst_abs = 0.0
    worst_rel = 0.0
    for case in _emotion_ferplus_cases(_static_input_shape(input_info)):
        inputs = {input_name: case}
        cpu_out = cpu.run(output_names, inputs)
        ggml_out = ggml.run(output_names, inputs)

        for got, expected in zip(ggml_out, cpu_out):
            diff = np.abs(got - expected)
            worst_abs = max(worst_abs, float(diff.max(initial=0.0)))
            worst_rel = max(
                worst_rel,
                float(
                    (
                        diff
                        / np.maximum(np.abs(expected), np.float32(1e-6))
                    ).max(initial=0.0)
                ),
            )
            np.testing.assert_allclose(
                got,
                expected,
                rtol=2e-3,
                atol=2e-4,
                err_msg=(
                    f"emotion-ferplus drift exceeded tolerance; "
                    f"worst_abs={worst_abs:.3e}, worst_rel={worst_rel:.3e}"
                ),
            )

    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_shufflenetv2_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("shufflenet-v2-10.onnx", _SHUFFLENETV2_MODEL_URL)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    rng = np.random.default_rng(0)
    inputs = {"input": rng.standard_normal((1, 3, 224, 224)).astype(np.float32)}

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)
    # assert_all_nodes_run_on_ggml(ggml) # FIXME: shufflelenet produces 5D intermediate that GGML does not like


@pytest.mark.integration
def test_tinystories_lstm_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("tinystories_lstm.onnx", _TINYSTORIES_LSTM_MODEL_URL)
    vocab_path = cached_model_path(
        "tinystories_lstm_vocab.txt", _TINYSTORIES_LSTM_VOCAB_URL
    )
    with vocab_path.open() as f:
        vocab = [line.strip() for line in f]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    sos_idx = word2idx["<SOS>"]
    pad_idx = word2idx["<PAD>"]
    unk_idx = word2idx["<UNK>"]

    import re

    def tokenize(text: str) -> list[str]:
        text = re.sub(r"([.,!?])", r" \1 ", text.lower())
        return text.split()

    seq_len = 50

    def build_input(token_ids: list[int]) -> np.ndarray:
        padded = token_ids + [pad_idx] * (seq_len - len(token_ids))
        if len(padded) > seq_len:
            padded = padded[-seq_len:]
        return np.array([padded], dtype=np.int64)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    input_name = cpu.get_inputs()[0].name
    output_names = [out.name for out in cpu.get_outputs()]

    # 1) Compare full logits on a fixed prompt.
    prompt_tokens = [sos_idx] + [
        word2idx.get(t, unk_idx) for t in tokenize("once upon a time")
    ]
    inputs = {input_name: build_input(prompt_tokens)}
    cpu_logits = cpu.run(output_names, inputs)[0]
    ggml_logits = ggml.run(output_names, inputs)[0]
    np.testing.assert_allclose(ggml_logits, cpu_logits, rtol=1e-3, atol=1e-3)

    # 2) Greedy autoregressive generation must agree for a few steps.
    cpu_seq = list(prompt_tokens)
    ggml_seq = list(prompt_tokens)
    for _ in range(5):
        cpu_out = cpu.run(output_names, {input_name: build_input(cpu_seq)})[0]
        ggml_out = ggml.run(output_names, {input_name: build_input(ggml_seq)})[0]
        cpu_pos = min(len(cpu_seq) - 1, seq_len - 1)
        ggml_pos = min(len(ggml_seq) - 1, seq_len - 1)
        cpu_next = int(np.argmax(cpu_out[0, cpu_pos, :]))
        ggml_next = int(np.argmax(ggml_out[0, ggml_pos, :]))
        assert cpu_next == ggml_next, (
            f"diverged at step {len(cpu_seq) - len(prompt_tokens)}: "
            f"cpu={cpu_next} ggml={ggml_next}"
        )
        cpu_seq.append(cpu_next)
        ggml_seq.append(ggml_next)


@pytest.mark.integration
def test_waifu2x_cunet_matches_cpu(ep_library: Path) -> None:
    raw_path = cached_model_path(
        "waifu2x_cunet_noise0_scale2x.onnx", _WAIFU2X_CUNET_MODEL_URL
    )
    model_path = _concretize_input_dims(raw_path, {"x": [1, 3, 128, 128]})

    inputs = {"x": _esperanto_flag(128, 128)}

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_waifu2x_swin_unet_matches_cpu(ep_library: Path) -> None:
    raw_path = cached_model_path(
        "waifu2x_swin_unet_art_noise0_scale2x.onnx",
        _WAIFU2X_SWIN_UNET_MODEL_URL,
    )
    model_path = _concretize_input_dims(raw_path, {"x": [1, 3, 128, 128]})

    inputs = {"x": _esperanto_flag(128, 128)}

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
def test_openwakeword_embedding_matches_cpu(ep_library: Path) -> None:
    raw_path = cached_model_path("embedding_model.onnx", _OPENWAKEWORD_EMBEDDING_URL)
    model_path = _concretize_input_dims(raw_path, {"input_1": [1, 76, 32, 1]})

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    rng = np.random.default_rng(0)
    inputs = {"input_1": rng.standard_normal((1, 76, 32, 1)).astype(np.float32)}

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_silero_vad_streaming_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("silero-vad.onnx", _SILERO_VAD_MODEL_URL)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    cpu_state = np.zeros((2, 1, 128), dtype=np.float32)
    ggml_state = np.zeros((2, 1, 128), dtype=np.float32)
    sample_rate = np.array(16000, dtype=np.int64)

    cpu_probs = []
    ggml_probs = []
    for chunk in _silero_vad_chunks():
        cpu_prob, cpu_state = cpu.run(
            None,
            {"input": chunk, "state": cpu_state, "sr": sample_rate},
        )
        ggml_prob, ggml_state = ggml.run(
            None,
            {"input": chunk, "state": ggml_state, "sr": sample_rate},
        )
        cpu_probs.append(cpu_prob)
        ggml_probs.append(ggml_prob)
        np.testing.assert_allclose(ggml_prob, cpu_prob, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ggml_state, cpu_state, rtol=1e-5, atol=2e-5)

    np.testing.assert_allclose(
        np.concatenate(ggml_probs, axis=0),
        np.concatenate(cpu_probs, axis=0),
        rtol=1e-5,
        atol=1e-6,
    )
    # ORT profiles GGML fused partitions under opaque fused-node ids rather than
    # the inner ONNX op names, so assert on provider activity plus the absence
    # of plain CPU Conv nodes in the encoder path.
    profile = end_profiling_profile(ggml)
    ggml_events = [
        event
        for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "GGMLExecutionProvider"
    ]
    assert ggml_events, "expected at least one GGML node event in Silero profile"
    cpu_plain_conv_events = [
        event
        for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "CPUExecutionProvider"
        and event.get("args", {}).get("op_name") == "Conv"
    ]
    assert not cpu_plain_conv_events, (
        f"found plain CPU Conv node events in Silero profile: {cpu_plain_conv_events}"
    )
