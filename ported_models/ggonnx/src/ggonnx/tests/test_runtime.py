#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

from test_support import (
    assert_all_nodes_run_on_ggml,
    assert_provider_does_not_run_ops,
    assert_provider_runs_any_node,
    assert_provider_runs_ops,
    cpu_session,
    end_profiling_profile,
    ggml_session,
    ensure_model,
    ensure_model_with_opset,
)


def build_single_add_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="add_0")
    graph = helper.make_graph([node], "single_add", [x, y], [z])
    return ensure_model(tmpdir, graph)


def build_dynamic_add_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "cols"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", "cols"])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", "cols"])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="add_dynamic")
    graph = helper.make_graph([node], "dynamic_add", [x, y], [z])
    return ensure_model(tmpdir, graph)


def build_broadcast_add_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "cols"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["cols"])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", "cols"])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="add_broadcast")
    graph = helper.make_graph([node], "broadcast_add", [x, y], [z])
    return ensure_model(tmpdir, graph)


def build_single_binary_model(tmpdir: Path, op_type: str) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(op_type, ["x", "y"], ["z"], name=f"{op_type.lower()}_0")
    graph = helper.make_graph([node], f"single_{op_type.lower()}", [x, y], [z])
    return ensure_model(tmpdir, graph)


def build_bidirectional_broadcast_add_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 2])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="add_bidirectional_bcast")
    graph = helper.make_graph([node], "bidirectional_broadcast_add", [x, y], [z])
    return ensure_model(tmpdir, graph)


def build_pow_model(tmpdir: Path, exponent: float) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    exp_init = helper.make_tensor("exp", TensorProto.FLOAT, [], [exponent])
    node = helper.make_node("Pow", ["x", "exp"], ["y"], name="pow_0")
    graph = helper.make_graph([node], "single_pow", [x], [y], initializer=[exp_init])
    return ensure_model(tmpdir, graph)


def build_cumsum_model(
    tmpdir: Path,
    *,
    shape,
    axis: int,
    exclusive: int = 0,
    reverse: int = 0,
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))
    axis_init = helper.make_tensor("axis", TensorProto.INT64, [], [axis])
    node = helper.make_node(
        "CumSum",
        ["x", "axis"],
        ["y"],
        name="cumsum_0",
        exclusive=exclusive,
        reverse=reverse,
    )
    graph = helper.make_graph([node], "single_cumsum", [x], [y], initializer=[axis_init])
    return ensure_model(tmpdir, graph)


def build_dynamic_mixed_binary_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "cols"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", "cols"])
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, ["batch", "cols"])
    m = helper.make_tensor_value_info("m", TensorProto.FLOAT, ["batch", "cols"])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, ["batch", "cols"])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", "cols"])
    nodes = [
        helper.make_node("Add", ["x", "y"], ["a"], name="add_0"),
        helper.make_node("Mul", ["a", "y"], ["m"], name="mul_0"),
        helper.make_node("Sub", ["m", "x"], ["s"], name="sub_0"),
        helper.make_node("Div", ["s", "y"], ["z"], name="div_0"),
    ]
    graph = helper.make_graph(nodes, "dynamic_mixed_binary", [x, y], [z], value_info=[a, m, s])
    return ensure_model(tmpdir, graph)


def build_single_gru_model(tmpdir: Path) -> Path:
    seq_length = 3
    batch_size = 2
    input_size = 4
    hidden_size = 3

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [seq_length, batch_size, input_size])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 3 * hidden_size, input_size])
    r = helper.make_tensor_value_info("r", TensorProto.FLOAT, [1, 3 * hidden_size, hidden_size])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 6 * hidden_size])
    initial_h = helper.make_tensor_value_info("initial_h", TensorProto.FLOAT, [1, batch_size, hidden_size])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [seq_length, 1, batch_size, hidden_size])
    y_h = helper.make_tensor_value_info("y_h", TensorProto.FLOAT, [1, batch_size, hidden_size])
    node = helper.make_node(
        "GRU",
        ["x", "w", "r", "b", "", "initial_h"],
        ["y", "y_h"],
        name="gru_0",
        hidden_size=hidden_size,
    )
    graph = helper.make_graph([node], "single_gru", [x, w, r, b, initial_h], [y, y_h])
    return ensure_model(tmpdir, graph)


def standard_inputs() -> dict[str, np.ndarray]:
    return {
        "x": np.array([[1.0, 2.0, 3.0], [4.5, -2.0, 7.0]], dtype=np.float32),
        "y": np.array([[0.5, -1.0, 8.0], [1.5, 3.0, -4.0]], dtype=np.float32),
    }


def assert_model_matches_cpu(model_path: Path, ep_library: Path, output_name: str, inputs: dict[str, np.ndarray]) -> None:
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    cpu_out = cpu.run([output_name], inputs)[0]
    ggml_out = ggml.run([output_name], inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def gru_inputs() -> dict[str, np.ndarray]:
    return {
        "x": np.array(
            [
                [[0.1, -0.2, 0.3, 0.5], [0.4, 0.0, -0.1, 0.2]],
                [[-0.3, 0.6, 0.2, -0.4], [0.7, -0.5, 0.1, 0.3]],
                [[0.2, 0.1, -0.6, 0.8], [-0.4, 0.9, 0.5, -0.2]],
            ],
            dtype=np.float32,
        ),
        "w": np.array(
            [
                [
                    [0.15, -0.20, 0.10, 0.05],
                    [-0.30, 0.25, 0.40, -0.10],
                    [0.20, 0.35, -0.15, 0.30],
                    [0.05, -0.10, 0.25, 0.40],
                    [-0.20, 0.30, -0.35, 0.15],
                    [0.45, -0.25, 0.05, -0.30],
                    [0.10, 0.15, -0.20, 0.35],
                    [-0.40, 0.05, 0.30, -0.15],
                    [0.25, -0.35, 0.20, 0.10],
                ]
            ],
            dtype=np.float32,
        ),
        "r": np.array(
            [
                [
                    [0.12, -0.18, 0.07],
                    [-0.22, 0.14, 0.31],
                    [0.28, -0.09, 0.16],
                    [0.05, 0.20, -0.11],
                    [-0.17, 0.26, 0.08],
                    [0.34, -0.21, 0.13],
                    [0.09, -0.04, 0.27],
                    [-0.29, 0.18, -0.07],
                    [0.16, 0.11, -0.24],
                ]
            ],
            dtype=np.float32,
        ),
        "b": np.array(
            [
                [
                    0.02,
                    -0.03,
                    0.01,
                    -0.04,
                    0.05,
                    -0.02,
                    0.03,
                    -0.01,
                    0.04,
                    -0.02,
                    0.01,
                    0.03,
                    0.05,
                    -0.04,
                    0.02,
                    -0.03,
                    0.02,
                    -0.01,
                ]
            ],
            dtype=np.float32,
        ),
        "initial_h": np.array(
            [[[0.05, -0.10, 0.15], [-0.20, 0.25, -0.05]]],
            dtype=np.float32,
        ),
    }


def test_single_add(suite_tmpdir, ep_library: Path) -> None:
    model_path = build_single_add_model(suite_tmpdir)
    assert_model_matches_cpu(model_path, ep_library, "z", standard_inputs())


def test_dynamic_add_cache_reuse(suite_tmpdir, ep_library: Path, debug_api) -> None:
    dynamic_model_path = build_dynamic_add_model(suite_tmpdir)
    cpu = cpu_session(dynamic_model_path)
    ggml = ggml_session(dynamic_model_path, ep_library)

    debug_api.reset()
    first_inputs = standard_inputs()
    cpu_out = cpu.run(["z"], first_inputs)[0]
    ggml_out = ggml.run(["z"], first_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 1

    same_shape_inputs = {
        "x": np.array([[9.0, 1.0, -2.0], [5.0, 6.0, 7.0]], dtype=np.float32),
        "y": np.array([[1.0, 1.5, 2.0], [0.0, -3.0, 10.0]], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], same_shape_inputs)[0]
    ggml_out = ggml.run(["z"], same_shape_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 1

    second_shape_inputs = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "y": np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], second_shape_inputs)[0]
    ggml_out = ggml.run(["z"], second_shape_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 2
    assert_all_nodes_run_on_ggml(ggml)


def test_broadcast_add_runs_on_ggml(suite_tmpdir, ep_library: Path, debug_api) -> None:
    model_path = build_broadcast_add_model(suite_tmpdir)
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    debug_api.reset()
    broadcast_inputs = {
        "x": np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32),
        "y": np.array([0.5, -1.0, 8.0], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], broadcast_inputs)[0]
    ggml_out = ggml.run(["z"], broadcast_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 1
    assert debug_api.last_graph_node_count() > 0
    assert debug_api.last_graph_output_count() == 1
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize("op_type", ["Add", "Sub", "Mul", "Div", "Max", "Min"])
def test_single_binary_ops(suite_tmpdir, ep_library: Path, op_type: str) -> None:
    model_path = build_single_binary_model(suite_tmpdir, op_type)
    assert_model_matches_cpu(model_path, ep_library, "z", standard_inputs())


def test_pow_square_runs_on_ggml(suite_tmpdir, ep_library: Path) -> None:
    model_path = build_pow_model(suite_tmpdir, 2.0)
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    inputs = {"x": standard_inputs()["x"]}
    cpu_out = cpu.run(["y"], inputs)[0]
    ggml_out = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def test_pow_non_square_falls_back_to_cpu(suite_tmpdir, ep_library: Path) -> None:
    model_path = build_pow_model(suite_tmpdir, 3.0)
    ggml = ggml_session(model_path, ep_library)
    inputs = {"x": standard_inputs()["x"]}
    ggml_out = ggml.run(["y"], inputs)[0]
    expected = np.power(inputs["x"], 3.0, dtype=np.float32)
    np.testing.assert_allclose(ggml_out, expected, rtol=1e-6, atol=1e-6)
    profile = end_profiling_profile(ggml)
    cpu_pow = [
        event for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "CPUExecutionProvider"
        and event.get("args", {}).get("op_name") == "Pow"
    ]
    ggml_pow = [
        event for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "GGMLExecutionProvider"
        and event.get("args", {}).get("op_name") == "Pow"
    ]
    assert cpu_pow, f"expected CPUExecutionProvider to run Pow, got profile: {profile}"
    assert not ggml_pow, f"unexpected GGMLExecutionProvider Pow events: {ggml_pow}"


def build_range_model(
    tmpdir: Path,
    *,
    start: float,
    limit: float,
    delta: float,
    dtype: int,
) -> Path:
    np_dtype = {
        TensorProto.FLOAT: np.float32,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
    }[dtype]
    n = max(int(np.ceil((limit - start) / delta)), 0)
    # Carry a dynamic scalar bias through the output so ORT's constant-folding
    # pass can't just fold the whole graph away — we want Range to actually run
    # on the EP at inference time.
    bias = helper.make_tensor_value_info("bias", dtype, [])
    y = helper.make_tensor_value_info("y", dtype, [n])
    start_init = helper.make_tensor("start", dtype, [], [np_dtype(start)])
    limit_init = helper.make_tensor("limit", dtype, [], [np_dtype(limit)])
    delta_init = helper.make_tensor("delta", dtype, [], [np_dtype(delta)])
    nodes = [
        helper.make_node("Range", ["start", "limit", "delta"], ["r"], name="range_0"),
        helper.make_node("Add", ["r", "bias"], ["y"], name="add_bias"),
    ]
    graph = helper.make_graph(
        nodes, "single_range", [bias], [y], initializer=[start_init, limit_init, delta_init]
    )
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "start,limit,delta,dtype",
    [
        (0.0, 10.0, 1.0, TensorProto.FLOAT),
        (1.5, 5.0, 0.5, TensorProto.FLOAT),
        (10.0, 2.0, -2.0, TensorProto.FLOAT),
    ],
)
def test_range_runs_on_ggml(suite_tmpdir, ep_library: Path, start, limit, delta, dtype) -> None:
    model_path = build_range_model(
        suite_tmpdir, start=start, limit=limit, delta=delta, dtype=dtype
    )
    np_dtype = {
        TensorProto.FLOAT: np.float32,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
    }[dtype]
    inputs = {"bias": np.array(0, dtype=np_dtype)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_array_equal(got, expected)
    assert_all_nodes_run_on_ggml(ggml)


def test_cumsum_trailing_axis_runs_on_ggml(suite_tmpdir, ep_library: Path) -> None:
    model_path = build_cumsum_model(suite_tmpdir, shape=(2, 3, 4), axis=-1)
    ggml = ggml_session(model_path, ep_library)
    inputs = {
        "x": np.array(
            [
                [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 1.5], [3.0, 0.0, -2.0, 5.0]],
                [[-1.0, 1.0, -1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [4.0, 3.0, 2.0, 1.0]],
            ],
            dtype=np.float32,
        ),
    }
    ggml_out = ggml.run(["y"], inputs)[0]
    expected = np.cumsum(inputs["x"], axis=-1, dtype=np.float32)
    np.testing.assert_allclose(ggml_out, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def test_cumsum_non_trailing_axis_falls_back_to_cpu(suite_tmpdir, ep_library: Path) -> None:
    model_path = build_cumsum_model(suite_tmpdir, shape=(2, 3, 4), axis=1)
    ggml = ggml_session(model_path, ep_library)
    inputs = {
        "x": np.array(
            [
                [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 1.5], [3.0, 0.0, -2.0, 5.0]],
                [[-1.0, 1.0, -1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [4.0, 3.0, 2.0, 1.0]],
            ],
            dtype=np.float32,
        ),
    }
    ggml_out = ggml.run(["y"], inputs)[0]
    expected = np.cumsum(inputs["x"], axis=1, dtype=np.float32)
    np.testing.assert_allclose(ggml_out, expected, rtol=1e-6, atol=1e-6)
    profile = end_profiling_profile(ggml)
    cpu_cumsum = [
        event for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "CPUExecutionProvider"
        and event.get("args", {}).get("op_name") == "CumSum"
    ]
    ggml_cumsum = [
        event for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "GGMLExecutionProvider"
        and event.get("args", {}).get("op_name") == "CumSum"
    ]
    assert cpu_cumsum, f"expected CPUExecutionProvider to run CumSum, got profile: {profile}"
    assert not ggml_cumsum, f"unexpected GGMLExecutionProvider CumSum events: {ggml_cumsum}"


def test_bidirectional_broadcast_add_runs_on_ggml(suite_tmpdir, ep_library: Path) -> None:
    # ONNX NumPy broadcast where neither operand matches the output shape
    # ([1,2] + [2,1] -> [2,2]). EmitElementwiseBinaryNode materializes the
    # missing side via ggml_repeat_4d, so the EP claims this rather than
    # falling back to CPU.
    model_path = build_bidirectional_broadcast_add_model(suite_tmpdir)
    ggml = ggml_session(model_path, ep_library)
    inputs = {
        "x": np.array([[1.0, 2.0]], dtype=np.float32),
        "y": np.array([[10.0], [20.0]], dtype=np.float32),
    }
    ggml_out = ggml.run(["z"], inputs)[0]
    expected = np.array([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32)
    np.testing.assert_allclose(ggml_out, expected, rtol=1e-6, atol=1e-6)
    profile = end_profiling_profile(ggml)
    cpu_add = [
        event for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "CPUExecutionProvider"
        and event.get("args", {}).get("op_name") == "Add"
    ]
    ggml_node = [
        event for event in profile
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "GGMLExecutionProvider"
    ]
    assert not cpu_add, f"expected GGML to handle Add, but CPU ran it: {cpu_add}"
    assert ggml_node, f"expected a GGMLExecutionProvider node, got profile: {profile}"


def build_scalar_broadcast_binary_model(tmpdir: Path, op_type: str, x_shape) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1] * len(x_shape))
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, list(x_shape))
    node = helper.make_node(op_type, ["x", "y"], ["z"], name=f"{op_type.lower()}_bcast")
    graph = helper.make_graph([node], f"bcast_{op_type.lower()}", [x, y], [z])
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize("op_type", ["Max", "Min"])
@pytest.mark.parametrize("x_shape", [(2, 3), (1, 4, 5, 6)])
def test_min_max_scalar_broadcast(suite_tmpdir, ep_library: Path, op_type, x_shape) -> None:
    model_path = build_scalar_broadcast_binary_model(suite_tmpdir, op_type, x_shape)
    rng = np.random.default_rng(31)
    inputs = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "y": np.array(-0.4, dtype=np.float32).reshape([1] * len(x_shape)),
    }
    assert_model_matches_cpu(model_path, ep_library, "z", inputs)


def test_dynamic_mixed_binary_cache_reuse(suite_tmpdir, ep_library: Path, debug_api) -> None:
    model_path = build_dynamic_mixed_binary_model(suite_tmpdir)
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    debug_api.reset()
    first_inputs = {
        "x": np.array([[1.0, 2.0, 4.0], [4.5, -2.0, 7.0]], dtype=np.float32),
        "y": np.array([[0.5, -1.0, 8.0], [1.5, 3.0, -4.0]], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], first_inputs)[0]
    ggml_out = ggml.run(["z"], first_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 1

    same_shape_inputs = {
        "x": np.array([[2.0, 5.0, 6.0], [8.0, 3.0, -1.0]], dtype=np.float32),
        "y": np.array([[4.0, 2.0, 3.0], [2.0, -4.0, 5.0]], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], same_shape_inputs)[0]
    ggml_out = ggml.run(["z"], same_shape_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 1

    second_shape_inputs = {
        "x": np.array([[4.0, 6.0], [8.0, 10.0], [12.0, 14.0]], dtype=np.float32),
        "y": np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], second_shape_inputs)[0]
    ggml_out = ggml.run(["z"], second_shape_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 2
    assert_all_nodes_run_on_ggml(ggml)


def build_single_unary_model(tmpdir: Path, op_type: str, shape=(2, 3), **attrs) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))
    node = helper.make_node(op_type, ["x"], ["y"], name=f"{op_type.lower()}_0", **attrs)
    graph = helper.make_graph([node], f"single_{op_type.lower()}", [x], [y])
    return ensure_model(tmpdir, graph)


def build_matmul_model(tmpdir: Path, a_shape, b_shape, out_shape) -> Path:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape))
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape))
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, list(out_shape))
    node = helper.make_node("MatMul", ["a", "b"], ["c"], name="matmul_0")
    graph = helper.make_graph([node], "single_matmul", [a, b], [c])
    return ensure_model(tmpdir, graph)


_UNARY_INPUT_OVERRIDES = {
    # Inputs suited to each op's valid domain.
    "Sqrt": np.array([[0.25, 1.0, 4.0], [9.0, 0.5, 2.0]], dtype=np.float32),
    "Log":  np.array([[0.5, 1.0, 2.0], [3.0, 4.0, 0.1]], dtype=np.float32),
}


@pytest.mark.parametrize(
    "op_type",
    ["Relu", "Sigmoid", "Tanh", "Neg", "Abs", "Sqrt", "Exp", "Log", "Erf", "Softplus", "Elu"],
)
def test_single_unary_ops(suite_tmpdir, ep_library: Path, op_type: str) -> None:
    model_path = build_single_unary_model(suite_tmpdir, op_type)
    default_x = np.array([[1.0, -2.0, 0.5], [-0.25, 3.0, -1.5]], dtype=np.float32)
    inputs = {"x": _UNARY_INPUT_OVERRIDES.get(op_type, default_x)}
    assert_model_matches_cpu(model_path, ep_library, "y", inputs)


@pytest.mark.parametrize(
    "op_type",
    ["Relu", "Sigmoid", "Tanh", "Neg", "Abs", "Sqrt", "Exp", "Log", "Softplus", "Elu", "Pow"],
)
def test_transpose_into_unary(suite_tmpdir, ep_library: Path, op_type: str) -> None:
    # Transpose produces a non-contiguous view; the unary/sqr CPU kernels assert
    # nb00 == sizeof(float). Regression guard for piper-style Transpose -> math
    # chains that would otherwise abort inside ggml.
    shape = [2, 3, 4]
    perm = [2, 0, 1]
    out_shape = [shape[p] for p in perm]
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)
    nodes = [helper.make_node("Transpose", ["x"], ["t"], name="transpose_0", perm=perm)]
    initializers = []
    if op_type == "Pow":
        import onnx
        exp_t = onnx.numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="exp")
        initializers.append(exp_t)
        nodes.append(helper.make_node("Pow", ["t", "exp"], ["y"], name="pow_0"))
    else:
        nodes.append(helper.make_node(op_type, ["t"], ["y"], name=f"{op_type.lower()}_0"))
    graph = helper.make_graph(nodes, f"transpose_{op_type.lower()}", [x_info], [y_info],
                              initializer=initializers)
    model_path = ensure_model(suite_tmpdir, graph)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, shape).astype(np.float32)
    assert_model_matches_cpu(model_path, ep_library, "y", {"x": x})


@pytest.mark.parametrize("op_type", ["Sqrt", "Sigmoid", "Pow", "Exp", "Abs"])
def test_transpose_ne0_one_into_unary(
    suite_tmpdir, ep_library: Path, op_type: str
) -> None:
    # Piper regression: a Transpose whose post-permute ne[0]==1 slips past
    # ggml_is_contiguous but still has nb[0] != type_size; unary CPU kernels
    # assert nb00 == sizeof(src0_t) unconditionally. The emitter must force
    # cont based on the stride, not the is_contiguous helper.
    import onnx
    shape = [1, 192, 1, 1]  # the exact ne pattern observed in piper
    perm = [1, 0, 2, 3]     # ne[0]=1, post-permute nb[0]=192*4
    out_shape = [shape[p] for p in perm]
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)
    initializers = []
    nodes = [helper.make_node("Transpose", ["x"], ["t"], perm=perm)]
    if op_type == "Pow":
        initializers.append(onnx.numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="exp"))
        nodes.append(helper.make_node("Pow", ["t", "exp"], ["y"]))
    else:
        nodes.append(helper.make_node(op_type, ["t"], ["y"]))
    graph = helper.make_graph(
        nodes, f"tr_ne0_{op_type}", [x_info], [y_info], initializer=initializers
    )
    model_path = ensure_model(suite_tmpdir, graph)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, shape).astype(np.float32)
    assert_model_matches_cpu(model_path, ep_library, "y", {"x": x})


@pytest.mark.parametrize(
    "op_type",
    ["Add", "Sub", "Mul", "Div"],
)
def test_transpose_into_binary(suite_tmpdir, ep_library: Path, op_type: str) -> None:
    # Transpose -> binary elementwise; some ggml binary kernels also assert
    # contiguous source.
    import onnx
    shape = [2, 3, 4]
    perm = [2, 0, 1]
    out_shape = [shape[p] for p in perm]
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)
    const = onnx.numpy_helper.from_array(
        np.ones(out_shape, dtype=np.float32), name="c"
    )
    nodes = [
        helper.make_node("Transpose", ["x"], ["t"], name="transpose_0", perm=perm),
        helper.make_node(op_type, ["t", "c"], ["y"], name=f"{op_type.lower()}_0"),
    ]
    graph = helper.make_graph(
        nodes, f"transpose_{op_type.lower()}", [x_info], [y_info], initializer=[const]
    )
    model_path = ensure_model(suite_tmpdir, graph)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, shape).astype(np.float32)
    assert_model_matches_cpu(model_path, ep_library, "y", {"x": x})


@pytest.mark.parametrize("op_type", ["Sqrt", "Sigmoid", "Pow", "Abs", "Log"])
def test_slice_into_unary(suite_tmpdir, ep_library: Path, op_type: str) -> None:
    # Slice produces a view; feeding it directly into a CPU unary kernel must
    # not fail on nb00 == sizeof.
    import onnx
    shape = [2, 6, 4]
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    initializers = [
        onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts"),
        onnx.numpy_helper.from_array(np.array([3], dtype=np.int64), name="ends"),
        onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes"),
        onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps"),
    ]
    nodes = [
        helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["s"]),
    ]
    if op_type == "Pow":
        initializers.append(onnx.numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="exp"))
        nodes.append(helper.make_node("Pow", ["s", "exp"], ["y"]))
    else:
        nodes.append(helper.make_node(op_type, ["s"], ["y"]))
    graph = helper.make_graph(nodes, f"slice_{op_type}", [x_info], [y_info], initializer=initializers)
    model_path = ensure_model(suite_tmpdir, graph)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, shape).astype(np.float32)
    assert_model_matches_cpu(model_path, ep_library, "y", {"x": x})


@pytest.mark.parametrize("op_type", ["Sqrt", "Sigmoid", "Pow"])
def test_transpose_reshape_into_unary(suite_tmpdir, ep_library: Path, op_type: str) -> None:
    import onnx
    shape = [2, 3, 4]
    perm = [2, 0, 1]  # -> [4, 2, 3]
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 6])
    initializers = [onnx.numpy_helper.from_array(np.array([4, 6], dtype=np.int64), name="shape")]
    nodes = [
        helper.make_node("Transpose", ["x"], ["t"], perm=perm),
        helper.make_node("Reshape", ["t", "shape"], ["r"]),
    ]
    if op_type == "Pow":
        initializers.append(onnx.numpy_helper.from_array(np.array(2.0, dtype=np.float32), name="exp"))
        nodes.append(helper.make_node("Pow", ["r", "exp"], ["y"]))
    else:
        nodes.append(helper.make_node(op_type, ["r"], ["y"]))
    graph = helper.make_graph(nodes, f"tr_{op_type}", [x_info], [y_info], initializer=initializers)
    model_path = ensure_model(suite_tmpdir, graph)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, shape).astype(np.float32)
    assert_model_matches_cpu(model_path, ep_library, "y", {"x": x})


def test_abs_float16(suite_tmpdir, ep_library: Path) -> None:
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 3])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 3])
    node = helper.make_node("Abs", ["x"], ["y"], name="abs_f16")
    graph = helper.make_graph([node], "abs_f16", [x_info], [y_info])
    model_path = ensure_model(suite_tmpdir, graph)

    x = np.array([[1.0, -2.0, 0.5], [-0.25, 3.0, -1.5]], dtype=np.float16)
    ggml = ggml_session(model_path, ep_library)
    (ggml_out,) = ggml.run(["y"], {"x": x})

    np.testing.assert_array_equal(ggml_out, np.abs(x))
    assert ggml_out.dtype == np.float16
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize(
    "op_type",
    ["Add", "Sub", "Mul", "Div", "PRelu", "Clip", "Concat",
     "Relu", "Sigmoid", "Tanh", "Neg", "Abs", "Sqrt", "Exp", "Log", "Softplus", "Elu"],
)
def test_float16_op(suite_tmpdir, ep_library: Path, op_type: str) -> None:
    rng = np.random.default_rng(42)
    shape = [2, 3]

    def f16(arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float16)

    if op_type in ("Add", "Sub", "Mul", "Div"):
        a_info = helper.make_tensor_value_info("a", TensorProto.FLOAT16, shape)
        b_info = helper.make_tensor_value_info("b", TensorProto.FLOAT16, shape)
        c_info = helper.make_tensor_value_info("c", TensorProto.FLOAT16, shape)
        node = helper.make_node(op_type, ["a", "b"], ["c"])
        graph = helper.make_graph([node], f"{op_type.lower()}_f16", [a_info, b_info], [c_info])
        model_path = ensure_model(suite_tmpdir, graph)
        a = f16(rng.uniform(0.5, 2.0, shape))
        b = f16(rng.uniform(0.5, 2.0, shape))
        inputs = {"a": a, "b": b}
        output_name = "c"
    elif op_type == "PRelu":
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)
        s_info = helper.make_tensor_value_info("slope", TensorProto.FLOAT16, [1])
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)
        node = helper.make_node("PRelu", ["x", "slope"], ["y"])
        graph = helper.make_graph([node], "prelu_f16", [x_info, s_info], [y_info])
        model_path = ensure_model(suite_tmpdir, graph)
        inputs = {"x": f16(rng.uniform(-2.0, 2.0, shape)), "slope": f16(np.array([0.1]))}
        output_name = "y"
    elif op_type == "Clip":
        import onnx
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)
        min_t = onnx.numpy_helper.from_array(np.array(-1.0, dtype=np.float16), name="clip_min")
        max_t = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float16), name="clip_max")
        node = helper.make_node("Clip", ["x", "clip_min", "clip_max"], ["y"])
        graph = helper.make_graph([node], "clip_f16", [x_info], [y_info], initializer=[min_t, max_t])
        model_path = ensure_model(suite_tmpdir, graph)
        inputs = {"x": f16(rng.uniform(-3.0, 3.0, shape))}
        output_name = "y"
    elif op_type == "Concat":
        a_info = helper.make_tensor_value_info("a", TensorProto.FLOAT16, shape)
        b_info = helper.make_tensor_value_info("b", TensorProto.FLOAT16, shape)
        c_info = helper.make_tensor_value_info("c", TensorProto.FLOAT16, [shape[0], shape[1] * 2])
        node = helper.make_node("Concat", ["a", "b"], ["c"], axis=1)
        graph = helper.make_graph([node], "concat_f16", [a_info, b_info], [c_info])
        model_path = ensure_model(suite_tmpdir, graph)
        inputs = {"a": f16(rng.standard_normal(shape)), "b": f16(rng.standard_normal(shape))}
        output_name = "c"
    else:
        # Unary ops
        overrides = {
            "Sqrt": f16(np.array([[0.25, 1.0, 4.0], [9.0, 0.5, 2.0]])),
            "Log":  f16(np.array([[0.5, 1.0, 2.0], [3.0, 4.0, 0.1]])),
        }
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)
        y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)
        node = helper.make_node(op_type, ["x"], ["y"])
        graph = helper.make_graph([node], f"{op_type.lower()}_f16", [x_info], [y_info])
        model_path = ensure_model(suite_tmpdir, graph)
        inputs = {"x": overrides.get(op_type, f16(np.array([[1.0, -2.0, 0.5], [-0.25, 3.0, -1.5]])))}
        output_name = "y"

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    cpu_out = cpu.run([output_name], inputs)[0]
    ggml_out = ggml.run([output_name], inputs)[0]

    assert ggml_out.dtype == np.float16
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-2, atol=1e-2)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize("alpha", [0.01, 0.2])
def test_single_leaky_relu(suite_tmpdir, ep_library: Path, alpha: float) -> None:
    model_path = build_single_unary_model(suite_tmpdir, "LeakyRelu", alpha=alpha)
    inputs = {"x": np.array([[1.0, -2.0, 0.5], [-0.25, 3.0, -1.5]], dtype=np.float32)}
    assert_model_matches_cpu(model_path, ep_library, "y", inputs)


@pytest.mark.parametrize(
    "x_shape,slope_shape",
    [
        ((2, 3), (1,)),          # scalar slope
        ((2, 3), (3,)),          # per-last-dim slope
        ((1, 4, 5, 5), (1, 4, 1, 1)),   # per-channel NCHW (arcface pattern)
        ((2, 4, 6, 6), (1, 4, 1, 1)),
    ],
)
def test_single_prelu(suite_tmpdir, ep_library: Path, x_shape, slope_shape) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    slope = helper.make_tensor_value_info("slope", TensorProto.FLOAT, list(slope_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x_shape))
    node = helper.make_node("PRelu", ["x", "slope"], ["y"], name="prelu_0")
    graph = helper.make_graph([node], "single_prelu", [x, slope], [y])
    model_path = ensure_model(suite_tmpdir, graph)
    rng = np.random.default_rng(3)
    inputs = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "slope": rng.standard_normal(slope_shape).astype(np.float32),
    }
    assert_model_matches_cpu(model_path, ep_library, "y", inputs)


@pytest.mark.parametrize("shape,axis", [((2, 4), -1), ((2, 4), 1), ((2, 3, 4), -1)])
def test_single_softmax(suite_tmpdir, ep_library: Path, shape, axis: int) -> None:
    model_path = build_single_unary_model(suite_tmpdir, "Softmax", shape=shape, axis=axis)
    rng = np.random.default_rng(0)
    inputs = {"x": rng.standard_normal(shape).astype(np.float32)}
    assert_model_matches_cpu(model_path, ep_library, "y", inputs)


@pytest.mark.parametrize(
    "a_shape,b_shape,out_shape",
    [
        ((3, 4), (4, 5), (3, 5)),
        ((2, 3, 4), (2, 4, 5), (2, 3, 5)),
        ((2, 2, 3, 4), (2, 2, 4, 5), (2, 2, 3, 5)),
    ],
)
def test_single_matmul(suite_tmpdir, ep_library: Path, a_shape, b_shape, out_shape) -> None:
    model_path = build_matmul_model(suite_tmpdir, a_shape, b_shape, out_shape)
    rng = np.random.default_rng(1)
    inputs = {
        "a": rng.standard_normal(a_shape).astype(np.float32),
        "b": rng.standard_normal(b_shape).astype(np.float32),
    }
    model = cpu_session(model_path)
    expected = model.run(["c"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["c"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


def build_conv_model(
    tmpdir: Path,
    x_shape,
    w_shape,
    y_shape,
    *,
    with_bias: bool = False,
    **attrs,
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, list(w_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    graph_inputs = [x, w]
    node_inputs = ["x", "w"]
    if with_bias:
        b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [w_shape[0]])
        graph_inputs.append(b)
        node_inputs.append("b")
    node = helper.make_node("Conv", node_inputs, ["y"], name="conv_0", **attrs)
    graph = helper.make_graph([node], "single_conv", graph_inputs, [y])
    return ensure_model(tmpdir, graph)


def _conv_inputs(x_shape, w_shape, with_bias: bool):
    rng = np.random.default_rng(2)
    inputs = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "w": rng.standard_normal(w_shape).astype(np.float32),
    }
    if with_bias:
        inputs["b"] = rng.standard_normal((w_shape[0],)).astype(np.float32)
    return inputs


# (x_shape, w_shape, y_shape, attrs, with_bias)
_CONV_CASES = [
    # 1D conv, no pad, stride 1: [N=1, C=2, W=8] * [4,2,3] -> [1,4,6]
    ((1, 2, 8), (4, 2, 3), (1, 4, 6),
     dict(kernel_shape=[3]), False),
    # 1D conv with bias and symmetric pad
    ((2, 3, 11), (5, 3, 3), (2, 5, 11),
     dict(kernel_shape=[3], pads=[1, 1]), True),
    # 1D conv stride 2
    ((1, 4, 12), (6, 4, 3), (1, 6, 5),
     dict(kernel_shape=[3], strides=[2]), True),
    # plain 3x3, no pad, stride 1 — [N=1, C=1, 5, 5] * [4,1,3,3] -> [1,4,3,3]
    ((1, 1, 5, 5), (4, 1, 3, 3), (1, 4, 3, 3),
     dict(kernel_shape=[3, 3]), False),
    # with bias
    ((1, 1, 5, 5), (4, 1, 3, 3), (1, 4, 3, 3),
     dict(kernel_shape=[3, 3]), True),
    # stride 2
    ((1, 3, 8, 8), (6, 3, 3, 3), (1, 6, 3, 3),
     dict(kernel_shape=[3, 3], strides=[2, 2]), True),
    # padding 1 (preserves spatial with 3x3 s=1)
    ((2, 3, 7, 7), (5, 3, 3, 3), (2, 5, 7, 7),
     dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1]), True),
    # LeNet-style: 1x28x28 with 5x5 kernel, 6 out channels, no pad, s=1 -> 1x6x24x24
    ((1, 1, 28, 28), (6, 1, 5, 5), (1, 6, 24, 24),
     dict(kernel_shape=[5, 5]), True),
    # LeNet conv2: 6->16, 5x5, no pad -> from 12x12 -> 8x8
    ((1, 6, 12, 12), (16, 6, 5, 5), (1, 16, 8, 8),
     dict(kernel_shape=[5, 5]), True),
    # dilation 2
    ((1, 2, 9, 9), (4, 2, 3, 3), (1, 4, 5, 5),
     dict(kernel_shape=[3, 3], dilations=[2, 2]), False),
    # asymmetric kernel
    ((1, 2, 6, 8), (3, 2, 1, 3), (1, 3, 6, 6),
     dict(kernel_shape=[1, 3]), True),
]


@pytest.mark.parametrize("x_shape,w_shape,y_shape,attrs,with_bias", _CONV_CASES)
def test_single_conv(suite_tmpdir, ep_library: Path, x_shape, w_shape, y_shape, attrs, with_bias) -> None:
    model_path = build_conv_model(
        suite_tmpdir, x_shape, w_shape, y_shape, with_bias=with_bias, **attrs
    )
    inputs = _conv_inputs(x_shape, w_shape, with_bias)
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert_all_nodes_run_on_ggml(ggml)


def build_pool_model(
    tmpdir: Path,
    op_type: str,
    x_shape,
    y_shape,
    **attrs,
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    node = helper.make_node(op_type, ["x"], ["y"], name=f"{op_type.lower()}_0", **attrs)
    graph = helper.make_graph([node], f"single_{op_type.lower()}", [x], [y])
    return ensure_model(tmpdir, graph)


def build_pad_model(tmpdir: Path, x_shape, y_shape, *, pads, mode="reflect", variant="legacy_attr") -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    if variant == "legacy_attr":
        node = helper.make_node("Pad", ["x"], ["y"], name="pad_0", pads=pads, mode=mode)
        graph = helper.make_graph([node], "single_pad_legacy", [x], [y])
        return ensure_model_with_opset(tmpdir, graph, 2)
    if variant == "input_pads":
        pads_init = helper.make_tensor("pads", TensorProto.INT64, [len(pads)], list(pads))
        node = helper.make_node("Pad", ["x", "pads"], ["y"], name="pad_0", mode=mode)
        graph = helper.make_graph([node], "single_pad_input_pads", [x], [y], initializer=[pads_init])
        return ensure_model_with_opset(tmpdir, graph, 13)
    raise ValueError(f"unsupported Pad variant: {variant}")


def build_instance_norm_model(tmpdir: Path, x_shape) -> Path:
    c = x_shape[1]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [c])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [c])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x_shape))
    node = helper.make_node(
        "InstanceNormalization",
        ["x", "scale", "bias"],
        ["y"],
        name="instance_norm_0",
        epsilon=1e-5,
    )
    graph = helper.make_graph([node], "single_instance_norm", [x, scale, bias], [y])
    return ensure_model(tmpdir, graph)


def build_upsample_model(
    tmpdir: Path, x_shape, y_shape, scales, *, variant="input_scales"
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    if variant == "legacy_attr":
        node = helper.make_node("Upsample", ["x"], ["y"], name="upsample_0", mode="nearest", scales=scales)
        graph = helper.make_graph([node], "single_upsample_legacy", [x], [y])
        return ensure_model_with_opset(tmpdir, graph, 7)
    if variant == "input_scales":
        scales_init = helper.make_tensor("scales", TensorProto.FLOAT, [4], list(scales))
        node = helper.make_node("Upsample", ["x", "scales"], ["y"], name="upsample_0", mode="nearest")
        graph = helper.make_graph([node], "single_upsample_input_scales", [x], [y], initializer=[scales_init])
        return ensure_model_with_opset(tmpdir, graph, 9)
    if variant == "resize":
        scales_init = helper.make_tensor("scales", TensorProto.FLOAT, [4], list(scales))
        node = helper.make_node(
            "Resize",
            ["x", "", "scales"],
            ["y"],
            name="resize_0",
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
        graph = helper.make_graph([node], "single_resize", [x], [y], initializer=[scales_init])
        return ensure_model_with_opset(tmpdir, graph, 13)
    raise ValueError(f"unsupported upsample variant: {variant}")


# (op_type, x_shape, y_shape, attrs)
_POOL_CASES = [
    # MaxPool, no pad, k=2 s=2
    ("MaxPool", (1, 3, 8, 8), (1, 3, 4, 4), dict(kernel_shape=[2, 2], strides=[2, 2])),
    # AveragePool, no pad, k=2 s=2
    ("AveragePool", (1, 3, 8, 8), (1, 3, 4, 4), dict(kernel_shape=[2, 2], strides=[2, 2])),
    # MaxPool with symmetric padding (padding matters less here since max ignores pad)
    ("MaxPool", (2, 4, 7, 7), (2, 4, 7, 7),
     dict(kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])),
    # AveragePool with count_include_pad=1 to match GGML's semantics
    ("AveragePool", (1, 2, 6, 6), (1, 2, 6, 6),
     dict(kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], count_include_pad=1)),
    # asymmetric kernel, stride 1
    ("MaxPool", (1, 2, 6, 8), (1, 2, 6, 6), dict(kernel_shape=[1, 3])),
    # auto_pad VALID
    ("AveragePool", (1, 3, 10, 10), (1, 3, 5, 5),
     dict(kernel_shape=[2, 2], strides=[2, 2], auto_pad="VALID")),
]


@pytest.mark.parametrize("op_type,x_shape,y_shape,attrs", _POOL_CASES)
def test_single_pool(suite_tmpdir, ep_library: Path, op_type, x_shape, y_shape, attrs) -> None:
    model_path = build_pool_model(suite_tmpdir, op_type, x_shape, y_shape, **attrs)
    rng = np.random.default_rng(9)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


_GLOBAL_POOL_CASES = [
    ("GlobalMaxPool", (1, 4, 7, 7), (1, 4, 1, 1)),
    ("GlobalAveragePool", (2, 3, 5, 8), (2, 3, 1, 1)),
]


@pytest.mark.parametrize("op_type,x_shape,y_shape", _GLOBAL_POOL_CASES)
def test_single_global_pool(suite_tmpdir, ep_library: Path, op_type, x_shape, y_shape) -> None:
    model_path = build_pool_model(suite_tmpdir, op_type, x_shape, y_shape)
    rng = np.random.default_rng(10)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize("variant", ["legacy_attr", "input_pads"])
@pytest.mark.parametrize(
    "x_shape,y_shape,pads",
    [
        # NCHW spatial pads (axes 2,3)
        ((1, 1, 4, 5), (1, 1, 6, 7), [0, 0, 1, 1, 0, 0, 1, 1]),
        ((1, 2, 6, 6), (1, 2, 14, 14), [0, 0, 4, 4, 0, 0, 4, 4]),
        # NHWC-style: pads on axes 1,2 — exercises the axis-agnostic path added
        # for waifu2x swin_unet. Reflect requires pads < source dim per axis.
        ((1, 5, 6, 3), (1, 7, 8, 3), [0, 1, 1, 0, 0, 1, 1, 0]),
        # Single non-innermost axis only — exercises the permute+cont rotation.
        ((2, 3, 4, 5), (2, 5, 4, 5), [0, 1, 0, 0, 0, 1, 0, 0]),
    ],
)
def test_single_pad_reflect(suite_tmpdir, ep_library: Path, variant, x_shape, y_shape, pads) -> None:
    model_path = build_pad_model(suite_tmpdir, x_shape, y_shape, pads=pads, variant=variant)
    rng = np.random.default_rng(11)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize(
    "x_shape,y_shape,pads",
    [
        # pure pad (waifu2x does not exercise this path, but it's the obvious case)
        ((1, 1, 4, 5), (1, 1, 6, 7), [0, 0, 1, 1, 0, 0, 1, 1]),
        # asymmetric pads
        ((1, 2, 3, 3), (1, 2, 5, 6), [0, 0, 1, 2, 0, 0, 1, 1]),
        # pure crop (waifu2x pattern: negative pads both sides)
        ((1, 3, 16, 16), (1, 3, 8, 8), [0, 0, -4, -4, 0, 0, -4, -4]),
        # mixed pad + crop per spatial dim
        ((1, 2, 6, 6), (1, 2, 5, 8), [0, 0, -1, 1, 0, 0, 0, 1]),
        # NHWC-style: pads on ONNX axes 1,2 (H,W) with axes 0,3 (B,C) untouched.
        # This is the waifu2x swin_unet window-partition pad pattern and drives
        # the axis-agnostic Pad support.
        ((1, 4, 4, 3), (1, 6, 5, 3), [0, 1, 1, 0, 0, 1, 0, 0]),
        # NHWC with end-only pads (no begin-side pad anywhere).
        ((2, 7, 7, 5), (2, 9, 9, 5), [0, 0, 0, 0, 0, 2, 2, 0]),
        # Single non-spatial axis pad — axis 0.
        ((1, 2, 3, 3), (3, 2, 3, 3), [1, 0, 0, 0, 1, 0, 0, 0]),
    ],
)
def test_single_pad_constant(suite_tmpdir, ep_library: Path, x_shape, y_shape, pads) -> None:
    model_path = build_pad_model(
        suite_tmpdir, x_shape, y_shape, pads=pads, mode="constant", variant="input_pads"
    )
    rng = np.random.default_rng(42)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def build_conv_transpose_model(
    tmpdir: Path, x_shape, w_shape, y_shape, *, with_bias: bool = False, **attrs
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, list(w_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    graph_inputs = [x, w]
    node_inputs = ["x", "w"]
    if with_bias:
        b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [w_shape[1]])
        graph_inputs.append(b)
        node_inputs.append("b")
    node = helper.make_node(
        "ConvTranspose", node_inputs, ["y"], name="conv_transpose_0", **attrs
    )
    graph = helper.make_graph([node], "single_conv_transpose", graph_inputs, [y])
    return ensure_model(tmpdir, graph)


# (x_shape, w_shape, y_shape, attrs, with_bias) — ONNX ConvTranspose weight
# layout is [IC, OC, KH, KW]; output = (in-1)*stride - 2*pad + kernel.
_CONV_TRANSPOSE_CASES = [
    # 2x upsample with 2x2 kernel, no pad (waifu2x conv2_up/conv3_up pattern)
    ((1, 4, 5, 5), (4, 8, 2, 2), (1, 8, 10, 10),
     dict(kernel_shape=[2, 2], strides=[2, 2]), True),
    # 2x upsample with 4x4 kernel, symmetric pad=3 (waifu2x conv_bottom pattern)
    ((1, 4, 6, 6), (4, 3, 4, 4), (1, 3, 12, 12),
     dict(kernel_shape=[4, 4], strides=[2, 2], pads=[3, 3, 3, 3]), True),
    # stride 1, no pad, no bias
    ((1, 2, 4, 4), (2, 3, 3, 3), (1, 3, 6, 6),
     dict(kernel_shape=[3, 3], strides=[1, 1]), False),
]


@pytest.mark.parametrize(
    "x_shape,w_shape,y_shape,attrs,with_bias", _CONV_TRANSPOSE_CASES
)
def test_single_conv_transpose(
    suite_tmpdir, ep_library: Path, x_shape, w_shape, y_shape, attrs, with_bias
) -> None:
    model_path = build_conv_transpose_model(
        suite_tmpdir, x_shape, w_shape, y_shape, with_bias=with_bias, **attrs
    )
    rng = np.random.default_rng(3)
    inputs = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "w": rng.standard_normal(w_shape).astype(np.float32),
    }
    if with_bias:
        inputs["b"] = rng.standard_normal((w_shape[1],)).astype(np.float32)
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert_all_nodes_run_on_ggml(ggml)


def build_expand_model(tmpdir: Path, x_shape, target_shape) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    # Broadcasting leading dims are allowed in ONNX Expand even when ranks differ.
    out_rank = max(len(x_shape), len(target_shape))
    padded_x = [1] * (out_rank - len(x_shape)) + list(x_shape)
    padded_t = [1] * (out_rank - len(target_shape)) + list(target_shape)
    y_shape = [max(a, b) for a, b in zip(padded_x, padded_t)]
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape)
    shape_init = helper.make_tensor(
        "shape", TensorProto.INT64, [len(target_shape)], list(target_shape)
    )
    node = helper.make_node("Expand", ["x", "shape"], ["y"], name="expand_0")
    graph = helper.make_graph(
        [node], "single_expand", [x], [y], initializer=[shape_init]
    )
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "x_shape,target_shape",
    [
        # SE-block broadcast from [N,C,1,1] to [N,C,H,W] — the waifu2x case
        ((1, 8, 1, 1), [1, 8, 16, 16]),
        # scalar broadcast to 4D
        ((1,), [2, 3, 4, 5]),
        # partial broadcast: 2D input expanded by prepending a new leading dim
        ((3, 1), [4, 3, 5]),
        # identity shape (no actual broadcasting needed)
        ((2, 3, 4, 5), [2, 3, 4, 5]),
    ],
)
def test_single_expand(suite_tmpdir, ep_library: Path, x_shape, target_shape) -> None:
    model_path = build_expand_model(suite_tmpdir, x_shape, target_shape)
    rng = np.random.default_rng(4)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize("x_shape", [(1, 3, 8, 8), (2, 4, 5, 7)])
def test_single_instance_normalization(suite_tmpdir, ep_library: Path, x_shape) -> None:
    model_path = build_instance_norm_model(suite_tmpdir, x_shape)
    c = x_shape[1]
    rng = np.random.default_rng(12)
    inputs = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "scale": rng.standard_normal((c,)).astype(np.float32),
        "bias": rng.standard_normal((c,)).astype(np.float32),
    }
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize("variant", ["legacy_attr", "input_scales", "resize"])
@pytest.mark.parametrize(
    "x_shape,y_shape,scales",
    [
        ((1, 3, 8, 8), (1, 3, 16, 16), [1.0, 1.0, 2.0, 2.0]),
        ((1, 2, 5, 7), (1, 2, 10, 14), [1.0, 1.0, 2.0, 2.0]),
    ],
)
def test_single_upsample_nearest(
    suite_tmpdir, ep_library: Path, variant, x_shape, y_shape, scales
) -> None:
    model_path = build_upsample_model(suite_tmpdir, x_shape, y_shape, scales, variant=variant)
    rng = np.random.default_rng(13)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def build_matmul_const_b_model(tmpdir: Path, a_shape, b_shape, out_shape, b_values) -> Path:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape))
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, list(out_shape))
    b_init = helper.make_tensor("b", TensorProto.FLOAT, list(b_shape), b_values.flatten().tolist())
    node = helper.make_node("MatMul", ["a", "b"], ["c"], name="matmul_const")
    graph = helper.make_graph([node], "matmul_const_b", [a], [c], initializer=[b_init])
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "a_shape,b_shape,out_shape",
    [
        ((3, 4), (4, 5), (3, 5)),
        ((2, 3, 4), (2, 4, 5), (2, 3, 5)),
    ],
)
def test_matmul_constant_b(suite_tmpdir, ep_library: Path, a_shape, b_shape, out_shape) -> None:
    rng = np.random.default_rng(7)
    b_values = rng.standard_normal(b_shape).astype(np.float32)
    model_path = build_matmul_const_b_model(suite_tmpdir, a_shape, b_shape, out_shape, b_values)
    inputs = {"a": rng.standard_normal(a_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["c"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["c"], inputs)[0]
    # Loose tolerance: on fp16-capable GPU backends (e.g. Vulkan) ggml's
    # pipeline_matmul_f32 uses float16_t internally for register caches, which
    # can drift by a few ×1e-3 vs CPU. `ep.ggonnx.matmul_precision=f32` is
    # accepted as a provider option but has no effect for the F32×F32 path
    # until llama.cpp grows a separate fp32-compute pipeline variant.
    np.testing.assert_allclose(got, expected, rtol=5e-3, atol=5e-3)
    assert_all_nodes_run_on_ggml(ggml)


def build_gemm_const_b_model(
    tmpdir: Path, a_shape, b_shape, y_shape, b_values, *, trans_b: int = 0
) -> Path:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    b_init = helper.make_tensor("b", TensorProto.FLOAT, list(b_shape), b_values.flatten().tolist())
    node = helper.make_node("Gemm", ["a", "b"], ["y"], name="gemm_const", transB=trans_b)
    graph = helper.make_graph([node], "gemm_const_b", [a], [y], initializer=[b_init])
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "a_shape,b_shape,y_shape,trans_b",
    [
        ((4, 5), (5, 3), (4, 3), 0),
        ((4, 5), (3, 5), (4, 3), 1),
    ],
)
def test_gemm_constant_b(suite_tmpdir, ep_library: Path, a_shape, b_shape, y_shape, trans_b) -> None:
    rng = np.random.default_rng(8)
    b_values = rng.standard_normal(b_shape).astype(np.float32)
    model_path = build_gemm_const_b_model(
        suite_tmpdir, a_shape, b_shape, y_shape, b_values, trans_b=trans_b
    )
    inputs = {"a": rng.standard_normal(a_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


def build_gemm_model(
    tmpdir: Path,
    a_shape,
    b_shape,
    y_shape,
    *,
    with_c: bool = False,
    c_shape=None,
    **attrs,
) -> Path:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape))
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    graph_inputs = [a, b]
    node_inputs = ["a", "b"]
    if with_c:
        c = helper.make_tensor_value_info("c", TensorProto.FLOAT, list(c_shape))
        graph_inputs.append(c)
        node_inputs.append("c")
    node = helper.make_node("Gemm", node_inputs, ["y"], name="gemm_0", **attrs)
    graph = helper.make_graph([node], "single_gemm", graph_inputs, [y])
    return ensure_model(tmpdir, graph)


# (a_shape, b_shape, y_shape, attrs, c_shape or None)
_GEMM_CASES = [
    # plain: A[M,K] @ B[K,N]
    ((4, 5), (5, 3), (4, 3), {}, None),
    # with bias vector [N]
    ((4, 5), (5, 3), (4, 3), {}, (3,)),
    # with bias [M,N]
    ((4, 5), (5, 3), (4, 3), {}, (4, 3)),
    # transA
    ((5, 4), (5, 3), (4, 3), dict(transA=1), (3,)),
    # transB
    ((4, 5), (3, 5), (4, 3), dict(transB=1), (3,)),
    # transA + transB
    ((5, 4), (3, 5), (4, 3), dict(transA=1, transB=1), (3,)),
    # alpha + beta
    ((4, 5), (5, 3), (4, 3), dict(alpha=0.5, beta=2.0), (3,)),
]


@pytest.mark.parametrize("a_shape,b_shape,y_shape,attrs,c_shape", _GEMM_CASES)
def test_single_gemm(suite_tmpdir, ep_library: Path, a_shape, b_shape, y_shape, attrs, c_shape) -> None:
    with_c = c_shape is not None
    model_path = build_gemm_model(
        suite_tmpdir, a_shape, b_shape, y_shape, with_c=with_c, c_shape=c_shape, **attrs
    )
    rng = np.random.default_rng(3)
    inputs = {
        "a": rng.standard_normal(a_shape).astype(np.float32),
        "b": rng.standard_normal(b_shape).astype(np.float32),
    }
    if with_c:
        inputs["c"] = rng.standard_normal(c_shape).astype(np.float32)
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


def build_reshape_model(tmpdir: Path, in_shape, out_shape) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))
    shape_init = helper.make_tensor(
        "shape", TensorProto.INT64, [len(out_shape)], list(out_shape)
    )
    node = helper.make_node("Reshape", ["x", "shape"], ["y"], name="reshape_0")
    graph = helper.make_graph(
        [node], "single_reshape", [x], [y], initializer=[shape_init]
    )
    return ensure_model(tmpdir, graph)


def build_squeeze_model(tmpdir: Path, x_shape, y_shape, axes, *, variant="input_axes") -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    if variant == "attribute_axes":
        node = helper.make_node("Squeeze", ["x"], ["y"], name="squeeze_0", axes=list(axes))
        graph = helper.make_graph([node], "single_squeeze_attr", [x], [y])
        return ensure_model_with_opset(tmpdir, graph, 11)
    if variant == "input_axes":
        axes_init = helper.make_tensor("axes", TensorProto.INT64, [len(axes)], list(axes))
        node = helper.make_node("Squeeze", ["x", "axes"], ["y"], name="squeeze_0")
        graph = helper.make_graph([node], "single_squeeze_input", [x], [y], initializer=[axes_init])
        return ensure_model_with_opset(tmpdir, graph, 13)
    raise ValueError(f"unsupported Squeeze variant: {variant}")


def build_unknown_rank_unsqueeze_model(tmpdir: Path, axes) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, None)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    axes_init = helper.make_tensor("axes", TensorProto.INT64, [len(axes)], list(axes))
    node = helper.make_node("Unsqueeze", ["x", "axes"], ["y"], name="unsqueeze_0")
    graph = helper.make_graph([node], "unknown_rank_unsqueeze", [x], [y], initializer=[axes_init])
    return ensure_model_with_opset(tmpdir, graph, 13)


@pytest.mark.parametrize(
    "in_shape,out_shape",
    [
        ((2, 3, 4), (6, 4)),
        ((1, 6, 12, 12), (1, 864)),
        ((24,), (2, 3, 4)),
        ((2, 3, 4, 5), (2, 60)),
    ],
)
def test_single_reshape(suite_tmpdir, ep_library: Path, in_shape, out_shape) -> None:
    model_path = build_reshape_model(suite_tmpdir, in_shape, out_shape)
    rng = np.random.default_rng(4)
    inputs = {"x": rng.standard_normal(in_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize("variant", ["attribute_axes", "input_axes"])
@pytest.mark.parametrize(
    "x_shape,y_shape,axes",
    [
        ((1, 3, 1, 5), (3, 5), [0, 2]),
        ((2, 1, 4, 1), (2, 4), [1, 3]),
    ],
)
def test_single_squeeze(
    suite_tmpdir, ep_library: Path, variant, x_shape, y_shape, axes
) -> None:
    model_path = build_squeeze_model(
        suite_tmpdir, x_shape, y_shape, axes, variant=variant
    )
    rng = np.random.default_rng(24)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def test_unsqueeze_with_unknown_input_rank_falls_back_gracefully(
    suite_tmpdir, ep_library: Path
) -> None:
    model_path = build_unknown_rank_unsqueeze_model(suite_tmpdir, [1])
    inputs = {"x": np.arange(128, dtype=np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_provider_does_not_run_ops(
        ggml, "GGMLExecutionProvider", {"Unsqueeze"}
    )


def build_window_shuffle_model(tmpdir: Path, in_shape, mid_shape, out_shape) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))
    s1 = helper.make_tensor("shape1", TensorProto.INT64, [len(mid_shape)], list(mid_shape))
    perm_out = list(mid_shape)
    perm_out[2], perm_out[3] = perm_out[3], perm_out[2]
    s2 = helper.make_tensor("shape2", TensorProto.INT64, [len(out_shape)], list(out_shape))
    nodes = [
        helper.make_node("Reshape", ["x", "shape1"], ["r1"], name="reshape_1"),
        helper.make_node("Transpose", ["r1"], ["t"], name="transpose_0", perm=[0, 1, 3, 2, 4, 5]),
        helper.make_node("Reshape", ["t", "shape2"], ["y"], name="reshape_2"),
    ]
    graph = helper.make_graph(
        nodes, "window_shuffle", [x], [y], initializer=[s1, s2]
    )
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "in_shape,mid_shape,out_shape",
    [
        # window_partition: [B, H, W, C] -> 6D -> [B*nw, M*M, C]
        ((1, 16, 16, 32), (1, 2, 8, 2, 8, 32), (4, 64, 32)),
        # window_reverse: [B*nw, M*M, C] -> 6D -> [B, H, W, C]
        ((4, 64, 32), (1, 2, 2, 8, 8, 32), (1, 16, 16, 32)),
        # Non-unit batch (B=2)
        ((2, 16, 16, 48), (2, 2, 8, 2, 8, 48), (8, 64, 48)),
        # Rectangular feature map H != W
        ((1, 16, 8, 32), (1, 2, 8, 1, 8, 32), (2, 64, 32)),
    ],
)
def test_single_window_shuffle(
    suite_tmpdir, ep_library: Path, in_shape, mid_shape, out_shape
) -> None:
    model_path = build_window_shuffle_model(suite_tmpdir, in_shape, mid_shape, out_shape)
    rng = np.random.default_rng(42)
    inputs = {"x": rng.standard_normal(in_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def build_slice_model(tmpdir: Path, x_shape, y_shape, starts, ends, axes=None, steps=None) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    inits = [
        helper.make_tensor("starts", TensorProto.INT64, [len(starts)], list(starts)),
        helper.make_tensor("ends", TensorProto.INT64, [len(ends)], list(ends)),
    ]
    node_inputs = ["x", "starts", "ends"]
    if axes is not None:
        inits.append(helper.make_tensor("axes", TensorProto.INT64, [len(axes)], list(axes)))
        node_inputs.append("axes")
    if steps is not None:
        if axes is None:
            inits.append(helper.make_tensor("axes", TensorProto.INT64, [len(steps)],
                                            list(range(len(steps)))))
            node_inputs.append("axes")
        inits.append(helper.make_tensor("steps", TensorProto.INT64, [len(steps)], list(steps)))
        node_inputs.append("steps")
    node = helper.make_node("Slice", node_inputs, ["y"], name="slice_0")
    graph = helper.make_graph([node], "single_slice", [x], [y], initializer=inits)
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "x_shape,y_shape,starts,ends,axes",
    [
        # Basic rank-4 slice on spatial dim
        ((1, 3, 8, 8), (1, 3, 4, 8), [0], [4], [2]),
        # Multi-axis slice
        ((1, 3, 8, 8), (1, 2, 4, 4), [1, 0, 2], [3, 4, 6], [1, 2, 3]),
        # Negative indexing
        ((1, 3, 8, 8), (1, 3, 8, 4), [-4], [8], [3]),
        # Rank 2
        ((6, 10), (3, 5), [1, 2], [4, 7], [0, 1]),
        # Rank 1
        ((12,), (4,), [2], [6], [0]),
        # Clamping: end beyond dim
        ((4, 5), (4, 3), [2], [99], [1]),
        # Default axes (none passed) — slice first len(starts) dims
        ((6, 8), (3, 8), [0], [3], None),
    ],
)
def test_single_slice(suite_tmpdir, ep_library: Path, x_shape, y_shape, starts, ends, axes) -> None:
    model_path = build_slice_model(suite_tmpdir, x_shape, y_shape, starts, ends, axes=axes)
    rng = np.random.default_rng(21)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize(
    "x_shape,starts,ends,axes,steps",
    [
        # step>1 on a non-last ONNX axis is view-expressible (supported on GGML).
        # Rank 2: step on axis 0.
        ((10, 4), [0], [10], [0], [2]),
        ((10, 4), [1], [10], [0], [3]),
        # Rank 3: step on axis 0 and on a middle axis.
        ((8, 6, 4), [0], [8], [0], [2]),
        ((8, 6, 4), [1], [6], [1], [2]),
        # Rank 4: step on a middle (NCHW spatial-row) axis.
        ((1, 3, 8, 8), [0], [8], [2], [2]),
        ((2, 4, 6, 8), [1, 0], [4, 6], [1, 2], [2, 3]),
        # End clamped beyond dim with stride.
        ((10, 4), [0], [99], [0], [3]),
    ],
)
def test_slice_step_gt_one_on_ggml(suite_tmpdir, ep_library: Path, x_shape, starts, ends, axes, steps) -> None:
    # Compute output shape via numpy so we don't have to spell it inline.
    rng = np.random.default_rng(33)
    x = rng.standard_normal(x_shape).astype(np.float32)
    sl = [slice(None)] * len(x_shape)
    for k, a in enumerate(axes):
        sl[a] = slice(starts[k], ends[k], steps[k])
    expected_shape = x[tuple(sl)].shape
    model_path = build_slice_model(
        suite_tmpdir, x_shape, expected_shape, starts, ends, axes=axes, steps=steps
    )
    cpu = cpu_session(model_path)
    inputs = {"x": x}
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize(
    "x_shape,starts,ends,axes,steps,reason",
    [
        # step>1 on the LAST ONNX axis = ggml axis 0; ggml_view forces nb[0]=type_size
        # so we can't express it as a view — must fall back to CPU.
        ((4, 10), [0], [10], [1], [2], "step on innermost axis"),
        # Negative step needs reverse / get_rows; not supported.
        ((10,), [9], [-11], [0], [-1], "negative step"),
        # step=0 is illegal in ONNX; we reject (CPU also rejects, but ggml support
        # predicate must not let it through).
    ],
)
def test_slice_unsupported_step_falls_back_cleanly(
    suite_tmpdir, ep_library: Path, x_shape, starts, ends, axes, steps, reason
) -> None:
    rng = np.random.default_rng(34)
    x = rng.standard_normal(x_shape).astype(np.float32)
    sl = [slice(None)] * len(x_shape)
    for k, a in enumerate(axes):
        sl[a] = slice(starts[k], ends[k], steps[k])
    out_shape = x[tuple(sl)].shape
    model_path = build_slice_model(
        suite_tmpdir, x_shape, out_shape, starts, ends, axes=axes, steps=steps
    )
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], {"x": x})[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], {"x": x})[0]
    # Numerically correct (CPU fallback) — we just don't assert ggml ran the node.
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def build_shape_folded_reshape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

    starts_i32 = helper.make_tensor("starts_i32", TensorProto.INT32, [1], [0])
    ends_i32 = helper.make_tensor("ends_i32", TensorProto.INT32, [1], [2])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Cast", ["shape"], ["shape_i32"], name="shape_cast", to=TensorProto.INT32),
        helper.make_node("Cast", ["starts_i32"], ["starts"], name="starts_cast", to=TensorProto.INT64),
        helper.make_node("Cast", ["ends_i32"], ["ends"], name="ends_cast", to=TensorProto.INT64),
        helper.make_node(
            "Slice",
            ["shape_i32", "starts", "ends", "axes", "steps"],
            ["shape_prefix_i32"],
            name="shape_slice",
        ),
        helper.make_node("Cast", ["shape_prefix_i32"], ["shape_prefix"], name="shape_prefix_cast", to=TensorProto.INT64),
        helper.make_node("Reshape", ["x", "shape_prefix"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(nodes, "shape_folded_reshape", [x], [y], initializer=[starts_i32, ends_i32, axes, steps])
    return ensure_model(tmpdir, graph)


def build_shape_concat_folded_reshape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

    starts0 = helper.make_tensor("starts0", TensorProto.INT64, [1], [0])
    ends0 = helper.make_tensor("ends0", TensorProto.INT64, [1], [1])
    starts1_i32 = helper.make_tensor("starts1_i32", TensorProto.INT32, [1], [1])
    ends1_i32 = helper.make_tensor("ends1_i32", TensorProto.INT32, [1], [2])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts0", "ends0", "axes", "steps"], ["dim0"], name="slice_dim0"),
        helper.make_node("Cast", ["shape"], ["shape_i32"], name="shape_cast", to=TensorProto.INT32),
        helper.make_node("Cast", ["starts1_i32"], ["starts1"], name="starts1_cast", to=TensorProto.INT64),
        helper.make_node("Cast", ["ends1_i32"], ["ends1"], name="ends1_cast", to=TensorProto.INT64),
        helper.make_node("Slice", ["shape_i32", "starts1", "ends1", "axes", "steps"], ["dim1_i32"], name="slice_dim1"),
        helper.make_node("Cast", ["dim1_i32"], ["dim1"], name="dim1_cast", to=TensorProto.INT64),
        helper.make_node("Concat", ["dim0", "dim1"], ["new_shape"], name="shape_concat", axis=0),
        helper.make_node("Reshape", ["x", "new_shape"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_concat_folded_reshape",
        [x],
        [y],
        initializer=[starts0, ends0, starts1_i32, ends1_i32, axes, steps],
    )
    return ensure_model(tmpdir, graph)


def build_shape_tile_folded_reshape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [2])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])
    repeats = helper.make_tensor("repeats", TensorProto.INT64, [1], [2])
    trim_starts = helper.make_tensor("trim_starts", TensorProto.INT64, [1], [0])
    trim_ends = helper.make_tensor("trim_ends", TensorProto.INT64, [1], [2])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts", "ends", "axes", "steps"], ["shape_copy"], name="slice_shape"),
        helper.make_node("Tile", ["shape_copy", "repeats"], ["tiled_shape"], name="tile_shape"),
        helper.make_node(
            "Slice",
            ["tiled_shape", "trim_starts", "trim_ends", "axes", "steps"],
            ["trimmed_shape"],
            name="trim_shape",
        ),
        helper.make_node("Reshape", ["x", "trimmed_shape"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_tile_folded_reshape",
        [x],
        [y],
        initializer=[starts, ends, axes, steps, repeats, trim_starts, trim_ends],
    )
    return ensure_model(tmpdir, graph)


def build_shape_gather_folded_reshape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4, 5])
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [20])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 5])

    indices = helper.make_tensor("indices", TensorProto.INT64, [2], [1, 2])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Gather", ["shape", "indices"], ["new_shape"], name="shape_gather", axis=0),
        helper.make_node("Reshape", ["data", "new_shape"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_gather_folded_reshape",
        [x, data],
        [y],
        initializer=[indices],
    )
    return ensure_model(tmpdir, graph)


def build_partial_shape_slice_folded_reshape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4, 5])
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [20])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 5])

    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [1])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [3])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts", "ends", "axes", "steps"], ["new_shape"], name="shape_slice"),
        helper.make_node("Reshape", ["data", "new_shape"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "partial_shape_slice_folded_reshape",
        [x, data],
        [y],
        initializer=[starts, ends, axes, steps],
    )
    return ensure_model(tmpdir, graph)


def build_partial_shape_unsqueeze_concat_folded_reshape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4, 5])
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [20])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 5])

    starts1 = helper.make_tensor("starts1", TensorProto.INT64, [1], [1])
    ends1 = helper.make_tensor("ends1", TensorProto.INT64, [1], [2])
    starts2 = helper.make_tensor("starts2", TensorProto.INT64, [1], [2])
    ends2 = helper.make_tensor("ends2", TensorProto.INT64, [1], [3])
    slice_axes = helper.make_tensor("slice_axes", TensorProto.INT64, [1], [0])
    slice_steps = helper.make_tensor("slice_steps", TensorProto.INT64, [1], [1])
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, [1], [0])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts1", "ends1", "slice_axes", "slice_steps"], ["dim1_vec"], name="slice_dim1"),
        helper.make_node("Squeeze", ["dim1_vec", "squeeze_axes"], ["dim1_scalar"], name="squeeze_dim1"),
        helper.make_node("Unsqueeze", ["dim1_scalar", "axes"], ["dim1"], name="unsqueeze_dim1"),
        helper.make_node("Slice", ["shape", "starts2", "ends2", "slice_axes", "slice_steps"], ["dim2"], name="slice_dim2"),
        helper.make_node("Concat", ["dim1", "dim2"], ["new_shape"], name="shape_concat", axis=0),
        helper.make_node("Reshape", ["data", "new_shape"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "partial_shape_unsqueeze_concat_folded_reshape",
        [x, data],
        [y],
        initializer=[starts1, ends1, starts2, ends2, slice_axes, slice_steps, squeeze_axes, axes],
    )
    return ensure_model(tmpdir, graph)


def build_partial_shape_constant_of_shape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4, 5])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 5])

    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [1])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [3])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])
    fill = helper.make_tensor("fill", TensorProto.FLOAT, [1], [0.5])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts", "ends", "axes", "steps"], ["new_shape"], name="shape_slice"),
        helper.make_node("ConstantOfShape", ["new_shape"], ["shape_bias"], name="shape_fill", value=fill),
        helper.make_node("Add", ["bias", "shape_bias"], ["y"], name="add_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "partial_shape_constant_of_shape",
        [x, bias],
        [y],
        initializer=[starts, ends, axes, steps],
    )
    return ensure_model(tmpdir, graph)


def build_loop_arange_folded_reshape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [1])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, [1], [0])
    cond = helper.make_tensor("loop_cond", TensorProto.BOOL, [], [True])
    start = helper.make_tensor("loop_start", TensorProto.INT32, [], [2])
    delta = helper.make_tensor("loop_delta", TensorProto.INT32, [], [1])

    body_in_iter = helper.make_tensor_value_info("i", TensorProto.INT64, [])
    body_in_cond = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    body_in_prev = helper.make_tensor_value_info("prev", TensorProto.INT32, [])
    body_out_cond = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    body_out_current = helper.make_tensor_value_info("current", TensorProto.INT32, [])
    body_out_range = helper.make_tensor_value_info("range", TensorProto.INT32, [])
    body_nodes = [
        helper.make_node("Add", ["prev", "loop_delta"], ["current"], name="loop_add"),
        helper.make_node("Identity", ["prev"], ["range"], name="loop_range"),
        helper.make_node("Identity", ["cond_in"], ["cond_out"], name="loop_cond_out"),
    ]
    body = helper.make_graph(
        body_nodes,
        "loop_body",
        [body_in_iter, body_in_cond, body_in_prev],
        [body_out_cond, body_out_current, body_out_range],
    )

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts", "ends", "axes", "steps"], ["trip_count_vec"], name="slice_trip_count"),
        helper.make_node("Squeeze", ["trip_count_vec", "squeeze_axes"], ["trip_count_i64"], name="squeeze_trip_count"),
        helper.make_node(
            "Loop",
            ["trip_count_i64", "loop_cond", "loop_start"],
            ["loop_final", "loop_range"],
            name="shape_loop",
            body=body,
        ),
        helper.make_node("Cast", ["loop_range"], ["reshape_shape"], name="cast_shape", to=TensorProto.INT64),
        helper.make_node("Reshape", ["x", "reshape_shape"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "loop_arange_folded_reshape",
        [x],
        [y],
        initializer=[starts, ends, axes, steps, squeeze_axes, cond, start, delta],
    )
    return ensure_model(tmpdir, graph)


def build_loop_arange_identity_alias_folded_reshape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [1])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, [1], [0])
    cond = helper.make_tensor("loop_cond", TensorProto.BOOL, [], [True])
    start = helper.make_tensor("loop_start", TensorProto.INT32, [], [2])
    delta = helper.make_tensor("loop_delta", TensorProto.INT32, [], [1])

    body_in_iter = helper.make_tensor_value_info("i", TensorProto.INT64, [])
    body_in_cond = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    body_in_prev = helper.make_tensor_value_info("prev", TensorProto.INT32, [])
    body_out_cond = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    body_out_current = helper.make_tensor_value_info("current", TensorProto.INT32, [])
    body_out_range = helper.make_tensor_value_info("range", TensorProto.INT32, [])
    body_nodes = [
        helper.make_node("Identity", ["prev"], ["prev_alias"], name="loop_prev_alias"),
        helper.make_node("Add", ["prev_alias", "loop_delta"], ["current"], name="loop_add"),
        helper.make_node("Identity", ["cond_in"], ["cond_out"], name="loop_cond_out"),
        helper.make_node("Identity", ["prev_alias"], ["range"], name="loop_range"),
    ]
    body = helper.make_graph(
        body_nodes,
        "loop_body_identity_alias",
        [body_in_iter, body_in_cond, body_in_prev],
        [body_out_cond, body_out_current, body_out_range],
    )

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts", "ends", "axes", "steps"], ["trip_count_vec"], name="slice_trip_count"),
        helper.make_node("Squeeze", ["trip_count_vec", "squeeze_axes"], ["trip_count_i64"], name="squeeze_trip_count"),
        helper.make_node(
            "Loop",
            ["trip_count_i64", "loop_cond", "loop_start"],
            ["loop_final", "loop_range"],
            name="shape_loop",
            body=body,
        ),
        helper.make_node("Cast", ["loop_range"], ["reshape_shape"], name="cast_shape", to=TensorProto.INT64),
        helper.make_node("Reshape", ["x", "reshape_shape"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "loop_arange_identity_alias_folded_reshape",
        [x],
        [y],
        initializer=[starts, ends, axes, steps, squeeze_axes, cond, start, delta],
    )
    return ensure_model(tmpdir, graph)


def build_partial_shape_loop_arange_folded_reshape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [1])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [2])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, [1], [0])
    cond = helper.make_tensor("loop_cond", TensorProto.BOOL, [], [True])
    start = helper.make_tensor("loop_start", TensorProto.INT32, [], [2])
    delta = helper.make_tensor("loop_delta", TensorProto.INT32, [], [1])

    body_in_iter = helper.make_tensor_value_info("i", TensorProto.INT64, [])
    body_in_cond = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    body_in_prev = helper.make_tensor_value_info("prev", TensorProto.INT32, [])
    body_out_cond = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    body_out_current = helper.make_tensor_value_info("current", TensorProto.INT32, [])
    body_out_range = helper.make_tensor_value_info("range", TensorProto.INT32, [])
    body_nodes = [
        helper.make_node("Add", ["prev", "loop_delta"], ["current"], name="loop_add"),
        helper.make_node("Identity", ["prev"], ["range"], name="loop_range"),
        helper.make_node("Identity", ["cond_in"], ["cond_out"], name="loop_cond_out"),
    ]
    body = helper.make_graph(
        body_nodes,
        "partial_shape_loop_body",
        [body_in_iter, body_in_cond, body_in_prev],
        [body_out_cond, body_out_current, body_out_range],
    )

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts", "ends", "axes", "steps"], ["trip_count_vec"], name="slice_trip_count"),
        helper.make_node("Squeeze", ["trip_count_vec", "squeeze_axes"], ["trip_count_i64"], name="squeeze_trip_count"),
        helper.make_node(
            "Loop",
            ["trip_count_i64", "loop_cond", "loop_start"],
            ["loop_final", "loop_range"],
            name="shape_loop",
            body=body,
        ),
        helper.make_node("Cast", ["loop_range"], ["reshape_shape"], name="cast_shape", to=TensorProto.INT64),
        helper.make_node("Reshape", ["x", "reshape_shape"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "partial_shape_loop_arange_folded_reshape",
        [x],
        [y],
        initializer=[starts, ends, axes, steps, squeeze_axes, cond, start, delta],
    )
    return ensure_model(tmpdir, graph)


def build_partial_shape_loop_arange_identity_alias_folded_reshape_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [1])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [2])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, [1], [0])
    cond = helper.make_tensor("loop_cond", TensorProto.BOOL, [], [True])
    start = helper.make_tensor("loop_start", TensorProto.INT32, [], [2])
    delta = helper.make_tensor("loop_delta", TensorProto.INT32, [], [1])

    body_in_iter = helper.make_tensor_value_info("i", TensorProto.INT64, [])
    body_in_cond = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    body_in_prev = helper.make_tensor_value_info("prev", TensorProto.INT32, [])
    body_out_cond = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    body_out_current = helper.make_tensor_value_info("current", TensorProto.INT32, [])
    body_out_range = helper.make_tensor_value_info("range", TensorProto.INT32, [])
    body_nodes = [
        helper.make_node("Identity", ["prev"], ["prev_alias"], name="loop_prev_alias"),
        helper.make_node("Add", ["prev_alias", "loop_delta"], ["current"], name="loop_add"),
        helper.make_node("Identity", ["cond_in"], ["cond_out"], name="loop_cond_out"),
        helper.make_node("Identity", ["prev_alias"], ["range"], name="loop_range"),
    ]
    body = helper.make_graph(
        body_nodes,
        "partial_shape_loop_body_identity_alias",
        [body_in_iter, body_in_cond, body_in_prev],
        [body_out_cond, body_out_current, body_out_range],
    )

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts", "ends", "axes", "steps"], ["trip_count_vec"], name="slice_trip_count"),
        helper.make_node("Squeeze", ["trip_count_vec", "squeeze_axes"], ["trip_count_i64"], name="squeeze_trip_count"),
        helper.make_node(
            "Loop",
            ["trip_count_i64", "loop_cond", "loop_start"],
            ["loop_final", "loop_range"],
            name="shape_loop",
            body=body,
        ),
        helper.make_node("Cast", ["loop_range"], ["reshape_shape"], name="cast_shape", to=TensorProto.INT64),
        helper.make_node("Reshape", ["x", "reshape_shape"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "partial_shape_loop_arange_identity_alias_folded_reshape",
        [x],
        [y],
        initializer=[starts, ends, axes, steps, squeeze_axes, cond, start, delta],
    )
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "builder",
    [
        build_shape_folded_reshape_model,
        build_shape_concat_folded_reshape_model,
        build_shape_tile_folded_reshape_model,
        build_shape_gather_folded_reshape_model,
        build_partial_shape_slice_folded_reshape_model,
        build_partial_shape_unsqueeze_concat_folded_reshape_model,
        build_partial_shape_constant_of_shape_model,
        build_loop_arange_folded_reshape_model,
        build_loop_arange_identity_alias_folded_reshape_model,
        build_partial_shape_loop_arange_folded_reshape_model,
        build_partial_shape_loop_arange_identity_alias_folded_reshape_model,
    ],
)
def test_compile_time_shape_subgraph_folding(suite_tmpdir, ep_library: Path, builder) -> None:
    model_path = builder(suite_tmpdir)
    rng = np.random.default_rng(23)
    if (
        builder is build_shape_gather_folded_reshape_model
        or builder is build_partial_shape_slice_folded_reshape_model
        or builder is build_partial_shape_unsqueeze_concat_folded_reshape_model
    ):
        inputs = {
            "x": rng.standard_normal((3, 4, 5)).astype(np.float32),
            "data": rng.standard_normal((20,)).astype(np.float32),
        }
    elif (
        builder is build_partial_shape_loop_arange_folded_reshape_model
        or builder is build_partial_shape_loop_arange_identity_alias_folded_reshape_model
    ):
        inputs = {"x": rng.standard_normal((1, 2, 3)).astype(np.float32)}
    elif builder is build_partial_shape_constant_of_shape_model:
        inputs = {
            "x": rng.standard_normal((3, 4, 5)).astype(np.float32),
            "bias": rng.standard_normal((4, 5)).astype(np.float32),
        }
    else:
        inputs = {"x": rng.standard_normal((2, 3)).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    if (
        builder is build_partial_shape_slice_folded_reshape_model
        or builder is build_partial_shape_unsqueeze_concat_folded_reshape_model
        or builder is build_partial_shape_loop_arange_folded_reshape_model
        or builder is build_partial_shape_loop_arange_identity_alias_folded_reshape_model
    ):
        profile = end_profiling_profile(ggml)
        ggml_events = [
            event
            for event in profile
            if event.get("cat") == "Node"
            and event.get("args", {}).get("provider") == "GGMLExecutionProvider"
        ]
        cpu_reshape_events = [
            event
            for event in profile
            if event.get("cat") == "Node"
            and event.get("args", {}).get("provider") == "CPUExecutionProvider"
            and event.get("args", {}).get("op_name") == "Reshape"
        ]
        assert ggml_events, "expected at least one GGML node event"
        assert not cpu_reshape_events, f"found CPU Reshape node events: {cpu_reshape_events}"
    elif builder is build_partial_shape_constant_of_shape_model:
        profile = end_profiling_profile(ggml)
        ggml_events = [
            event
            for event in profile
            if event.get("cat") == "Node"
            and event.get("args", {}).get("provider") == "GGMLExecutionProvider"
        ]
        cpu_add_events = [
            event
            for event in profile
            if event.get("cat") == "Node"
            and event.get("args", {}).get("provider") == "CPUExecutionProvider"
            and event.get("args", {}).get("op_name") == "Add"
        ]
        assert ggml_events, "expected at least one GGML node event"
        assert not cpu_add_events, f"found CPU Add node events: {cpu_add_events}"
    else:
        assert_all_nodes_run_on_ggml(ggml)


def build_batchnorm_model(tmpdir: Path, x_shape, *, epsilon=1e-5) -> Path:
    c = x_shape[1]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [c])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [c])
    mean = helper.make_tensor_value_info("mean", TensorProto.FLOAT, [c])
    var = helper.make_tensor_value_info("var", TensorProto.FLOAT, [c])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x_shape))
    node = helper.make_node(
        "BatchNormalization",
        ["x", "scale", "bias", "mean", "var"],
        ["y"],
        name="bn_0",
        epsilon=epsilon,
    )
    graph = helper.make_graph([node], "single_bn", [x, scale, bias, mean, var], [y])
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "x_shape",
    [(1, 4, 6, 6), (2, 8, 3, 3), (1, 3, 16, 16), (1, 512), (2, 16), (2, 4, 5)],
)
def test_single_batchnorm(suite_tmpdir, ep_library: Path, x_shape) -> None:
    model_path = build_batchnorm_model(suite_tmpdir, x_shape)
    rng = np.random.default_rng(22)
    c = x_shape[1]
    inputs = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "scale": rng.standard_normal((c,)).astype(np.float32),
        "bias": rng.standard_normal((c,)).astype(np.float32),
        "mean": rng.standard_normal((c,)).astype(np.float32),
        # var must be positive for sqrt to be meaningful
        "var": np.abs(rng.standard_normal((c,)).astype(np.float32)) + 0.1,
    }
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


def build_single_lstm_model(tmpdir: Path) -> Path:
    seq_length = 3
    batch_size = 2
    input_size = 4
    hidden_size = 3

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [seq_length, batch_size, input_size])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 4 * hidden_size, input_size])
    r = helper.make_tensor_value_info("r", TensorProto.FLOAT, [1, 4 * hidden_size, hidden_size])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 8 * hidden_size])
    initial_h = helper.make_tensor_value_info("initial_h", TensorProto.FLOAT, [1, batch_size, hidden_size])
    initial_c = helper.make_tensor_value_info("initial_c", TensorProto.FLOAT, [1, batch_size, hidden_size])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [seq_length, 1, batch_size, hidden_size])
    y_h = helper.make_tensor_value_info("y_h", TensorProto.FLOAT, [1, batch_size, hidden_size])
    y_c = helper.make_tensor_value_info("y_c", TensorProto.FLOAT, [1, batch_size, hidden_size])
    node = helper.make_node(
        "LSTM",
        ["x", "w", "r", "b", "", "initial_h", "initial_c"],
        ["y", "y_h", "y_c"],
        name="lstm_0",
        hidden_size=hidden_size,
    )
    graph = helper.make_graph(
        [node], "single_lstm", [x, w, r, b, initial_h, initial_c], [y, y_h, y_c]
    )
    return ensure_model(tmpdir, graph)


def lstm_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    hidden_size = 3
    input_size = 4
    batch_size = 2
    seq_length = 3
    return {
        "x": rng.standard_normal((seq_length, batch_size, input_size)).astype(np.float32) * 0.5,
        "w": rng.standard_normal((1, 4 * hidden_size, input_size)).astype(np.float32) * 0.3,
        "r": rng.standard_normal((1, 4 * hidden_size, hidden_size)).astype(np.float32) * 0.3,
        "b": rng.standard_normal((1, 8 * hidden_size)).astype(np.float32) * 0.1,
        "initial_h": rng.standard_normal((1, batch_size, hidden_size)).astype(np.float32) * 0.2,
        "initial_c": rng.standard_normal((1, batch_size, hidden_size)).astype(np.float32) * 0.2,
    }


def test_single_lstm_matches_cpu(suite_tmpdir, ep_library: Path) -> None:
    model_path = build_single_lstm_model(suite_tmpdir)
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    inputs = lstm_inputs()
    cpu_y, cpu_y_h, cpu_y_c = cpu.run(["y", "y_h", "y_c"], inputs)
    ggml_y, ggml_y_h, ggml_y_c = ggml.run(["y", "y_h", "y_c"], inputs)
    np.testing.assert_allclose(ggml_y, cpu_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ggml_y_h, cpu_y_h, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ggml_y_c, cpu_y_c, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


def build_depth_to_space_model(
    tmpdir: Path, x_shape, y_shape, blocksize: int, mode: str
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    node = helper.make_node(
        "DepthToSpace",
        ["x"],
        ["y"],
        name="d2s_0",
        blocksize=blocksize,
        mode=mode,
    )
    graph = helper.make_graph([node], "single_depth_to_space", [x], [y])
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize("mode", ["DCR", "CRD"])
@pytest.mark.parametrize(
    "x_shape,blocksize",
    [
        ((1, 4, 2, 3), 2),
        ((1, 12, 5, 7), 2),
        ((1, 9, 4, 4), 3),
        ((1, 8, 1, 1), 2),
        ((2, 4, 2, 3), 2),
        ((3, 12, 5, 7), 2),
        ((4, 9, 4, 4), 3),
    ],
)
def test_single_depth_to_space(
    suite_tmpdir, ep_library: Path, x_shape, blocksize, mode
) -> None:
    n, c, h, w = x_shape
    y_shape = (n, c // (blocksize * blocksize), h * blocksize, w * blocksize)
    model_path = build_depth_to_space_model(
        suite_tmpdir, x_shape, y_shape, blocksize, mode
    )
    rng = np.random.default_rng(57)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def test_single_gru_matches_cpu(suite_tmpdir, ep_library: Path) -> None:
    model_path = build_single_gru_model(suite_tmpdir)
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    inputs = gru_inputs()
    cpu_y, cpu_y_h = cpu.run(["y", "y_h"], inputs)
    ggml_y, ggml_y_h = ggml.run(["y", "y_h"], inputs)
    np.testing.assert_allclose(ggml_y, cpu_y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ggml_y_h, cpu_y_h, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


# ----------------------------------------------------------------------------
# Fusion tests: patterns that individually contain rank-5 intermediates
# unsupported by ggml. The fact that these run at all (no ORT CPU fallback)
# proves the fusion detector found + rewired them. `assert_all_nodes_run_on_ggml`
# pins that property down.
# ----------------------------------------------------------------------------


def build_qkv_split_model(tmpdir: Path, batch: int, tokens: int, heads: int, head_dim: int) -> Path:
    # Packed QKV: [B, T, 3*H*D] -> Reshape [B, T, 3, H, D]
    #   -> Transpose(perm=[2,0,3,1,4])
    #   -> 3x Gather(axis=0, scalar index) -> [B, H, T, D] per Q/K/V.
    # Each Gather output is squared and summed into `y` so the test exercises
    # real use of the three tensors.
    x_shape = [batch, tokens, 3 * heads * head_dim]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, heads, tokens, head_dim])

    rshape = helper.make_tensor(
        "rshape", TensorProto.INT64, [5], [batch, tokens, 3, heads, head_dim]
    )
    idx0 = helper.make_tensor("idx0", TensorProto.INT64, [], [0])
    idx1 = helper.make_tensor("idx1", TensorProto.INT64, [], [1])
    idx2 = helper.make_tensor("idx2", TensorProto.INT64, [], [2])

    nodes = [
        helper.make_node("Reshape", ["x", "rshape"], ["r_out"], name="r"),
        helper.make_node("Transpose", ["r_out"], ["t_out"], name="t", perm=[2, 0, 3, 1, 4]),
        helper.make_node("Gather", ["t_out", "idx0"], ["q"], name="gq", axis=0),
        helper.make_node("Gather", ["t_out", "idx1"], ["k"], name="gk", axis=0),
        helper.make_node("Gather", ["t_out", "idx2"], ["v"], name="gv", axis=0),
        # Combine to force all three to be used.
        helper.make_node("Mul", ["q", "q"], ["q2"], name="mq"),
        helper.make_node("Mul", ["k", "k"], ["k2"], name="mk"),
        helper.make_node("Mul", ["v", "v"], ["v2"], name="mv"),
        helper.make_node("Add", ["q2", "k2"], ["qk"], name="aqk"),
        helper.make_node("Add", ["qk", "v2"], ["y"], name="ayv"),
    ]
    graph = helper.make_graph(
        nodes,
        "qkv_split_fusion",
        [x],
        [y],
        initializer=[rshape, idx0, idx1, idx2],
    )
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "batch,tokens,heads,head_dim",
    [
        (2, 9, 4, 8),
        (1, 16, 6, 16),
    ],
)
def test_qkv_split_fusion_matches_cpu(
    suite_tmpdir, ep_library: Path, batch: int, tokens: int, heads: int, head_dim: int
) -> None:
    model_path = build_qkv_split_model(suite_tmpdir, batch, tokens, heads, head_dim)
    rng = np.random.default_rng(7)
    inputs = {
        "x": rng.standard_normal((batch, tokens, 3 * heads * head_dim)).astype(np.float32)
    }
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    # Without the fusion the rank-5 Transpose output would force ORT to bounce
    # to CPU — this assertion pins the fusion behaviour.
    assert_all_nodes_run_on_ggml(ggml)


def build_window_mask_add_model(
    tmpdir: Path, batch: int, num_windows: int, heads: int, tokens: int
) -> Path:
    # Score tensor: [B*nw, H, T, T].
    # Fuse:
    #   Reshape -> [B, nw, H, T, T]   (rank 5, unsupported by ggml alone)
    #   Add with mask [1, nw, 1, T, T]
    #   Reshape back -> [B*nw, H, T, T]
    # The detector drops the leading size-1 ONNX axis of the mask, leaving
    # [nw, 1, T, T] which broadcasts into [B*nw, H, T, T] natively.
    bnw = batch * num_windows
    x_shape = [bnw, heads, tokens, tokens]
    y_shape = x_shape

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape)

    split_shape = helper.make_tensor(
        "split_shape", TensorProto.INT64, [5], [batch, num_windows, heads, tokens, tokens]
    )
    merge_shape = helper.make_tensor("merge_shape", TensorProto.INT64, [4], y_shape)
    rng = np.random.default_rng(123)
    mask_np = rng.standard_normal((1, num_windows, 1, tokens, tokens)).astype(np.float32)
    mask_init = helper.make_tensor(
        "mask", TensorProto.FLOAT, list(mask_np.shape), mask_np.flatten().tolist()
    )

    nodes = [
        helper.make_node("Reshape", ["x", "split_shape"], ["x5"], name="r1"),
        helper.make_node("Add", ["x5", "mask"], ["x5m"], name="a"),
        helper.make_node("Reshape", ["x5m", "merge_shape"], ["y"], name="r2"),
    ]
    graph = helper.make_graph(
        nodes,
        "window_mask_add_fusion",
        [x],
        [y],
        initializer=[split_shape, merge_shape, mask_init],
    )
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "batch,num_windows,heads,tokens",
    [
        (2, 5, 3, 4),
        (3, 7, 6, 6),
    ],
)
def test_window_mask_add_fusion_matches_cpu(
    suite_tmpdir, ep_library: Path, batch: int, num_windows: int, heads: int, tokens: int
) -> None:
    model_path = build_window_mask_add_model(suite_tmpdir, batch, num_windows, heads, tokens)
    rng = np.random.default_rng(11)
    inputs = {
        "x": rng.standard_normal((batch * num_windows, heads, tokens, tokens)).astype(np.float32)
    }
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def test_concat_integer_runs_on_ggml(suite_tmpdir, ep_library: Path) -> None:
    # Verifies INT32/INT64/INT8 Concat paths: tensors flow through GGML (not
    # folded or rejected), match CPU values, and retain their integer dtype.
    # All dtypes share one session — init dominates per-case runtime.
    cases = [
        ("i64", np.int64, TensorProto.INT64),
        ("i32", np.int32, TensorProto.INT32),
        ("i8",  np.int8,  TensorProto.INT8),
    ]
    input_infos, output_infos, nodes = [], [], []
    for tag, _dt, tt in cases:
        input_infos.append(helper.make_tensor_value_info(f"a_{tag}", tt, [2, 3]))
        input_infos.append(helper.make_tensor_value_info(f"b_{tag}", tt, [2, 4]))
        output_infos.append(helper.make_tensor_value_info(f"c_{tag}", tt, [2, 7]))
        nodes.append(helper.make_node("Concat", [f"a_{tag}", f"b_{tag}"],
                                      [f"c_{tag}"], name=f"concat_{tag}", axis=1))
    graph = helper.make_graph(nodes, "concat_int_all", input_infos, output_infos)
    model_path = ensure_model(suite_tmpdir, graph)

    rng = np.random.default_rng(0)
    inputs = {}
    for tag, dt, _tt in cases:
        info = np.iinfo(dt)
        lo = max(info.min, -1000)
        hi = min(info.max, 1000)
        inputs[f"a_{tag}"] = rng.integers(lo, hi, size=(2, 3)).astype(dt)
        inputs[f"b_{tag}"] = rng.integers(lo, hi, size=(2, 4)).astype(dt)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    output_names = [f"c_{c[0]}" for c in cases]
    cpu_outs = cpu.run(output_names, inputs)
    ggml_outs = ggml.run(output_names, inputs)
    for (tag, dt, _tt), cpu_out, ggml_out in zip(cases, cpu_outs, ggml_outs):
        assert ggml_out.dtype == dt, tag
        np.testing.assert_array_equal(ggml_out, cpu_out, err_msg=tag)
    assert_all_nodes_run_on_ggml(ggml)


def test_shape_derived_concat_to_constantofshape(suite_tmpdir, ep_library: Path) -> None:
    # Mirrors the TinyStories-LSTM pattern that previously crashed: an INT64
    # Concat whose middle input is a Shape->Gather->Unsqueeze chain bound to
    # the runtime batch dim. The Concat must defer to CPU because its partition
    # has no subgraph input carrying the bound source tensor; downstream
    # ConstantOfShape must see the real batch size.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, "batch", 3])

    gather_idx = helper.make_tensor("gather_idx", TensorProto.INT64, [], [0])
    unsq_axes = helper.make_tensor("unsq_axes", TensorProto.INT64, [1], [0])
    two = helper.make_tensor("two", TensorProto.INT64, [1], [2])
    three = helper.make_tensor("three", TensorProto.INT64, [1], [3])

    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"], name="shape0"),
        helper.make_node("Gather", ["x_shape", "gather_idx"], ["batch_scalar"],
                         name="gather0", axis=0),
        helper.make_node("Unsqueeze", ["batch_scalar", "unsq_axes"], ["batch_1d"],
                         name="unsq0"),
        helper.make_node("Concat", ["two", "batch_1d", "three"], ["out_shape"],
                         name="shape_concat", axis=0),
        helper.make_node("ConstantOfShape", ["out_shape"], ["y"], name="zeros"),
    ]
    graph = helper.make_graph(
        nodes, "shape_derived_concat", [x], [y],
        initializer=[gather_idx, unsq_axes, two, three],
    )
    model_path = ensure_model(suite_tmpdir, graph)

    inputs = {"x": np.zeros((5, 4), dtype=np.float32)}
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    expected = cpu.run(["y"], inputs)[0]
    got = ggml.run(["y"], inputs)[0]
    assert got.shape == (2, 5, 3)
    np.testing.assert_array_equal(got, expected)
    # The shape-bound Gather->Unsqueeze value depends on runtime input `x`.
    # Concat must not run inside a GGML partition that lacks `x` as a real
    # subgraph input, otherwise the shape-bound constant upload will fail.
    profile = end_profiling_profile(ggml)
    node_events = [e for e in profile if e.get("cat") == "Node"]
    cpu_concats = [
        e for e in node_events
        if e.get("args", {}).get("provider") == "CPUExecutionProvider"
        and e.get("args", {}).get("op_name") == "Concat"
    ]
    assert cpu_concats, f"expected Concat to stay on CPU, got profile: {profile}"


def test_cast_runs_on_ggml(suite_tmpdir, ep_library: Path) -> None:
    # All supported (src,dst) pairs in one graph so we pay session init once.
    cases = [
        ("f32_to_i32", np.float32, TensorProto.FLOAT,   np.int32,   TensorProto.INT32),
        ("i32_to_f32", np.int32,   TensorProto.INT32,   np.float32, TensorProto.FLOAT),
        ("f32_to_f16", np.float32, TensorProto.FLOAT,   np.float16, TensorProto.FLOAT16),
        ("f16_to_f32", np.float16, TensorProto.FLOAT16, np.float32, TensorProto.FLOAT),
        ("f32_to_f32", np.float32, TensorProto.FLOAT,   np.float32, TensorProto.FLOAT),
    ]

    input_infos, output_infos, nodes = [], [], []
    for name, _src_np, src_type, _dst_np, dst_type in cases:
        input_infos.append(helper.make_tensor_value_info(f"x_{name}", src_type, [2, 3]))
        output_infos.append(helper.make_tensor_value_info(f"y_{name}", dst_type, [2, 3]))
        nodes.append(helper.make_node("Cast", [f"x_{name}"], [f"y_{name}"],
                                      name=f"cast_{name}", to=dst_type))
    graph = helper.make_graph(nodes, "cast_all", input_infos, output_infos)
    model_path = ensure_model(suite_tmpdir, graph)

    rng = np.random.default_rng(0)
    inputs = {}
    for name, src_np, _src_type, _dst_np, _dst_type in cases:
        if np.issubdtype(src_np, np.integer):
            info = np.iinfo(src_np)
            lo = max(info.min, -100)
            hi = min(info.max, 100)
            inputs[f"x_{name}"] = rng.integers(lo, hi, size=(2, 3)).astype(src_np)
        else:
            inputs[f"x_{name}"] = (rng.standard_normal((2, 3)) * 50.0).astype(src_np)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    output_names = [f"y_{c[0]}" for c in cases]
    cpu_outs = cpu.run(output_names, inputs)
    ggml_outs = ggml.run(output_names, inputs)
    for (name, _src_np, _src_type, dst_np, _dst_type), cpu_out, ggml_out in zip(
            cases, cpu_outs, ggml_outs):
        assert ggml_out.dtype == dst_np, name
        np.testing.assert_array_equal(ggml_out, cpu_out, err_msg=name)
    assert_all_nodes_run_on_ggml(ggml)


def build_pad_then_slice_model(tmpdir: Path) -> Path:
    # Pad ([2, 3, 4] -> [2, 3, 6] with pads on last axis), Slice the padded
    # output along axis 1 (start=0, end=2). Validates the Slice support
    # predicate consults the resolved shape even after the (folded) Pad.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2, 6])

    pads = helper.make_tensor(
        "pads", TensorProto.INT64, [6], [0, 0, 1, 0, 0, 1])
    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [2])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Pad", ["x", "pads"], ["padded"], name="pad_0", mode="constant"),
        helper.make_node(
            "Slice", ["padded", "starts", "ends", "axes", "steps"], ["y"], name="slice_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "pad_then_slice",
        [x],
        [y],
        initializer=[pads, starts, ends, axes, steps],
    )
    return ensure_model(tmpdir, graph)


def build_shape_concat_reshape_then_slice_model(tmpdir: Path) -> Path:
    # Same shape-math pattern as build_shape_concat_folded_reshape_model but
    # followed by a Slice on the reshape output. ORT's shape inference doesn't
    # reliably propagate shape through a Concat-of-Shape-of-input used as a
    # Reshape target, so the Reshape output's dims land as symbolic in ORT's
    # view. meta_eval folds the Shape+Concat chain into a concrete int64
    # tensor, and the new PropagateInferredShapes pass then makes the Reshape
    # output's shape available to downstream Slice via ResolveShape.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])

    starts0 = helper.make_tensor("starts0", TensorProto.INT64, [1], [0])
    ends0 = helper.make_tensor("ends0", TensorProto.INT64, [1], [1])
    starts1_i32 = helper.make_tensor("starts1_i32", TensorProto.INT32, [1], [1])
    ends1_i32 = helper.make_tensor("ends1_i32", TensorProto.INT32, [1], [2])
    axes0 = helper.make_tensor("axes0", TensorProto.INT64, [1], [0])
    steps0 = helper.make_tensor("steps0", TensorProto.INT64, [1], [1])

    # Outer Slice on the reshape output: take the first row ([1, 3] from [2, 3]).
    outer_starts = helper.make_tensor("outer_starts", TensorProto.INT64, [1], [0])
    outer_ends = helper.make_tensor("outer_ends", TensorProto.INT64, [1], [1])
    outer_axes = helper.make_tensor("outer_axes", TensorProto.INT64, [1], [0])
    outer_steps = helper.make_tensor("outer_steps", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts0", "ends0", "axes0", "steps0"], ["dim0"], name="slice_dim0"),
        helper.make_node("Cast", ["shape"], ["shape_i32"], name="shape_cast", to=TensorProto.INT32),
        helper.make_node("Cast", ["starts1_i32"], ["starts1"], name="starts1_cast", to=TensorProto.INT64),
        helper.make_node("Cast", ["ends1_i32"], ["ends1"], name="ends1_cast", to=TensorProto.INT64),
        helper.make_node("Slice", ["shape_i32", "starts1", "ends1", "axes0", "steps0"], ["dim1_i32"], name="slice_dim1"),
        helper.make_node("Cast", ["dim1_i32"], ["dim1"], name="dim1_cast", to=TensorProto.INT64),
        helper.make_node("Concat", ["dim0", "dim1"], ["new_shape"], name="shape_concat", axis=0),
        helper.make_node("Reshape", ["x", "new_shape"], ["reshaped"], name="reshape_0"),
        helper.make_node(
            "Slice",
            ["reshaped", "outer_starts", "outer_ends", "outer_axes", "outer_steps"],
            ["y"],
            name="slice_data",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_concat_reshape_then_slice",
        [x],
        [y],
        initializer=[
            starts0, ends0, starts1_i32, ends1_i32, axes0, steps0,
            outer_starts, outer_ends, outer_axes, outer_steps,
        ],
    )
    return ensure_model(tmpdir, graph)


def test_pad_then_slice_runs_on_ggml(suite_tmpdir, ep_library: Path) -> None:
    model_path = build_pad_then_slice_model(suite_tmpdir)
    rng = np.random.default_rng(17)
    inputs = {"x": rng.standard_normal((2, 3, 4)).astype(np.float32)}
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    expected = cpu.run(["y"], inputs)[0]
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    # GGML fuses its partition into a single event named GGMLExecutionProvider_*
    # so a per-op "Slice" event won't appear. The invariant we care about: the
    # data-path Slice must not fall back to the CPU EP, and at least one GGML
    # partition runs. This test is a regression guard for the migrated
    # predicate more than a shape-propagation test; pair with the Shape+
    # Reshape+Slice test below.
    profile = end_profiling_profile(ggml)
    node_events = [e for e in profile if e.get("cat") == "Node"]
    providers = {e.get("args", {}).get("provider") for e in node_events}
    cpu_slices = [
        e for e in node_events
        if e.get("args", {}).get("provider") == "CPUExecutionProvider"
        and e.get("args", {}).get("op_name") == "Slice"
    ]
    assert not cpu_slices, f"Slice fell back to CPU: {cpu_slices}"
    assert "GGMLExecutionProvider" in providers, "no GGML partition executed"


def test_shape_concat_reshape_then_slice_runs_on_ggml(
        suite_tmpdir, ep_library: Path) -> None:
    model_path = build_shape_concat_reshape_then_slice_model(suite_tmpdir)
    rng = np.random.default_rng(31)
    inputs = {"x": rng.standard_normal((2, 3)).astype(np.float32)}
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    expected = cpu.run(["y"], inputs)[0]
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    # The outer Slice consumes a Reshape output whose shape ORT leaves
    # symbolic (the Reshape target comes from Shape+Slice+Concat). Shape
    # propagation through Reshape gives meta_eval a concrete shape, and
    # ResolveShape lets the Slice support predicate accept — so the outer
    # data-path Slice must not fall back to CPU.
    profile = end_profiling_profile(ggml)
    node_events = [e for e in profile if e.get("cat") == "Node"]
    providers = {e.get("args", {}).get("provider") for e in node_events}
    # The inner shape-math Slices on int64 shape tensors are expected to be
    # folded into initializers by meta_eval, so only the outer data-path Slice
    # would ever appear. None of them should execute on CPU.
    cpu_slices = [
        e for e in node_events
        if e.get("args", {}).get("provider") == "CPUExecutionProvider"
        and e.get("args", {}).get("op_name") == "Slice"
    ]
    assert not cpu_slices, f"Slice fell back to CPU: {cpu_slices}"
    assert "GGMLExecutionProvider" in providers, "no GGML partition executed"


def test_shape_reduce_round_ceil_range_reshape_runs_on_ggml(
        suite_tmpdir, ep_library: Path) -> None:
    model_path = build_shape_reduce_round_ceil_range_reshape_model(suite_tmpdir)
    rng = np.random.default_rng(41)
    inputs = {"x": rng.standard_normal((2, 3)).astype(np.float32)}
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    expected = cpu.run(["y"], inputs)[0]
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    profile = end_profiling_profile(ggml)
    cpu_shape_math = [
        e for e in profile
        if e.get("cat") == "Node"
        and e.get("args", {}).get("provider") == "CPUExecutionProvider"
        and e.get("args", {}).get("op_name") in {"ReduceMin", "Ceil", "Round", "Range"}
    ]
    assert not cpu_shape_math, f"shape math fell back to CPU: {cpu_shape_math}"
    providers = {
        e.get("args", {}).get("provider")
        for e in profile
        if e.get("cat") == "Node"
    }
    assert "GGMLExecutionProvider" in providers, "no GGML partition executed"


def build_shape_concat_reshape_then_slice_chain_model(tmpdir: Path) -> Path:
    # Like build_shape_concat_reshape_then_slice_model, but the Reshape output
    # is first Transposed and then Sliced — exercises the Phase-2 Transpose
    # shape rule so that downstream Slice can resolve the input shape via
    # meta_eval even after the Reshape output's dims are symbolic in ORT.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 1])

    starts0 = helper.make_tensor("starts0", TensorProto.INT64, [1], [0])
    ends0 = helper.make_tensor("ends0", TensorProto.INT64, [1], [1])
    starts1_i32 = helper.make_tensor("starts1_i32", TensorProto.INT32, [1], [1])
    ends1_i32 = helper.make_tensor("ends1_i32", TensorProto.INT32, [1], [2])
    axes0 = helper.make_tensor("axes0", TensorProto.INT64, [1], [0])
    steps0 = helper.make_tensor("steps0", TensorProto.INT64, [1], [1])

    outer_starts = helper.make_tensor("outer_starts", TensorProto.INT64, [1], [0])
    outer_ends = helper.make_tensor("outer_ends", TensorProto.INT64, [1], [1])
    outer_axes = helper.make_tensor("outer_axes", TensorProto.INT64, [1], [1])
    outer_steps = helper.make_tensor("outer_steps", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts0", "ends0", "axes0", "steps0"], ["dim0"], name="slice_dim0"),
        helper.make_node("Cast", ["shape"], ["shape_i32"], name="shape_cast", to=TensorProto.INT32),
        helper.make_node("Cast", ["starts1_i32"], ["starts1"], name="starts1_cast", to=TensorProto.INT64),
        helper.make_node("Cast", ["ends1_i32"], ["ends1"], name="ends1_cast", to=TensorProto.INT64),
        helper.make_node("Slice", ["shape_i32", "starts1", "ends1", "axes0", "steps0"], ["dim1_i32"], name="slice_dim1"),
        helper.make_node("Cast", ["dim1_i32"], ["dim1"], name="dim1_cast", to=TensorProto.INT64),
        helper.make_node("Concat", ["dim0", "dim1"], ["new_shape"], name="shape_concat", axis=0),
        helper.make_node("Reshape", ["x", "new_shape"], ["reshaped"], name="reshape_0"),
        helper.make_node("Transpose", ["reshaped"], ["transposed"], name="transpose_0", perm=[1, 0]),
        helper.make_node(
            "Slice",
            ["transposed", "outer_starts", "outer_ends", "outer_axes", "outer_steps"],
            ["y"],
            name="slice_data",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_concat_reshape_transpose_slice",
        [x],
        [y],
        initializer=[
            starts0, ends0, starts1_i32, ends1_i32, axes0, steps0,
            outer_starts, outer_ends, outer_axes, outer_steps,
        ],
    )
    return ensure_model(tmpdir, graph)


def build_shape_concat_reshape_then_expand_model(tmpdir: Path) -> Path:
    # Like build_shape_concat_reshape_then_slice_model, but the symbolic
    # Reshape output feeds an Expand. ORT often leaves the Reshape result
    # shape symbolic because the target comes from Shape+Slice+Concat.
    # meta_eval must propagate that shape so Expand can accept.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4, 3])

    starts0 = helper.make_tensor("starts0", TensorProto.INT64, [1], [0])
    ends0 = helper.make_tensor("ends0", TensorProto.INT64, [1], [1])
    starts1_i32 = helper.make_tensor("starts1_i32", TensorProto.INT32, [1], [1])
    ends1_i32 = helper.make_tensor("ends1_i32", TensorProto.INT32, [1], [2])
    axes0 = helper.make_tensor("axes0", TensorProto.INT64, [1], [0])
    steps0 = helper.make_tensor("steps0", TensorProto.INT64, [1], [1])
    one = helper.make_tensor("one", TensorProto.INT64, [1], [1])
    expand_shape = helper.make_tensor("expand_shape", TensorProto.INT64, [3], [2, 4, 3])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts0", "ends0", "axes0", "steps0"], ["dim0"], name="slice_dim0"),
        helper.make_node("Cast", ["shape"], ["shape_i32"], name="shape_cast", to=TensorProto.INT32),
        helper.make_node("Cast", ["starts1_i32"], ["starts1"], name="starts1_cast", to=TensorProto.INT64),
        helper.make_node("Cast", ["ends1_i32"], ["ends1"], name="ends1_cast", to=TensorProto.INT64),
        helper.make_node("Slice", ["shape_i32", "starts1", "ends1", "axes0", "steps0"], ["dim1_i32"], name="slice_dim1"),
        helper.make_node("Cast", ["dim1_i32"], ["dim1"], name="dim1_cast", to=TensorProto.INT64),
        helper.make_node("Concat", ["dim0", "one", "dim1"], ["new_shape"], name="shape_concat", axis=0),
        helper.make_node("Reshape", ["x", "new_shape"], ["reshaped"], name="reshape_0"),
        helper.make_node("Expand", ["reshaped", "expand_shape"], ["y"], name="expand_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_concat_reshape_expand",
        [x],
        [y],
        initializer=[starts0, ends0, starts1_i32, ends1_i32, axes0, steps0, one, expand_shape],
    )
    return ensure_model(tmpdir, graph)


def build_shape_reduce_round_ceil_range_reshape_model(tmpdir: Path) -> Path:
    # Compile-time shape math with ops that meta_eval did not previously fold:
    # ReduceMin, Ceil, Round, and Range. All of these are used only to build
    # the Reshape target, so none should execute on CPU at runtime.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3])

    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    one_f = helper.make_tensor("one_f", TensorProto.FLOAT, [], [1.0])
    one_i64_start = helper.make_tensor("one_i64_start", TensorProto.INT64, [], [1])
    one_i64 = helper.make_tensor("one_i64", TensorProto.INT64, [], [1])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("ReduceMin", ["shape"], ["min_dim"], name="reduce_min_0", keepdims=0, axes=[0]),
        helper.make_node("Cast", ["min_dim"], ["min_dim_f"], name="cast_to_float", to=TensorProto.FLOAT),
        helper.make_node("Div", ["min_dim_f", "one_f"], ["half_dim"], name="div_0"),
        helper.make_node("Ceil", ["half_dim"], ["ceil_dim"], name="ceil_0"),
        helper.make_node("Round", ["ceil_dim"], ["round_dim"], name="round_0"),
        helper.make_node("Cast", ["round_dim"], ["limit_i64"], name="cast_to_i64", to=TensorProto.INT64),
        helper.make_node("Range", ["one_i64_start", "limit_i64", "one_i64"], ["range_dim"], name="range_0"),
        helper.make_node("Concat", ["range_dim", "shape"], ["target_shape"], name="concat_0", axis=0),
        helper.make_node("Reshape", ["x", "target_shape"], ["y"], name="reshape_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_reduce_round_ceil_range_reshape",
        [x],
        [y],
        initializer=[axes, one_f, one_i64_start, one_i64],
    )
    return ensure_model(tmpdir, graph)


def build_shape_concat_reshape_transpose_split_unsqueeze_model(tmpdir: Path) -> Path:
    # Reshape target comes from Shape+Slice+Div+Concat, which ORT often leaves
    # symbolic. meta_eval must propagate that shape through Reshape and
    # Transpose, then infer all Split outputs so the downstream
    # Split->Squeeze->Unsqueeze branch stays on GGML.
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 12])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1, 1, 3])

    starts0 = helper.make_tensor("starts0", TensorProto.INT64, [1], [0])
    ends0 = helper.make_tensor("ends0", TensorProto.INT64, [1], [1])
    starts1 = helper.make_tensor("starts1", TensorProto.INT64, [1], [1])
    ends1 = helper.make_tensor("ends1", TensorProto.INT64, [1], [2])
    axes0 = helper.make_tensor("axes0", TensorProto.INT64, [1], [0])
    steps0 = helper.make_tensor("steps0", TensorProto.INT64, [1], [1])
    one = helper.make_tensor("one", TensorProto.INT64, [1], [1])
    four = helper.make_tensor("four", TensorProto.INT64, [1], [4])
    split_sizes = helper.make_tensor("split_sizes", TensorProto.INT64, [4], [1, 1, 1, 1])
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, [1], [0])
    unsqueeze_axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [1])

    nodes = [
        helper.make_node("Shape", ["x"], ["shape"], name="shape_0"),
        helper.make_node("Slice", ["shape", "starts0", "ends0", "axes0", "steps0"], ["dim0"], name="slice_dim0"),
        helper.make_node("Slice", ["shape", "starts1", "ends1", "axes0", "steps0"], ["dim1"], name="slice_dim1"),
        helper.make_node("Div", ["dim1", "four"], ["dim1_div4"], name="div_dim1"),
        helper.make_node("Concat", ["dim0", "four", "one", "dim1_div4"], ["target_shape"], name="shape_concat", axis=0),
        helper.make_node("Reshape", ["x", "target_shape"], ["reshaped"], name="reshape_0"),
        helper.make_node("Transpose", ["reshaped"], ["transposed"], name="transpose_0", perm=[1, 0, 2, 3]),
        helper.make_node("Split", ["transposed", "split_sizes"], ["s0", "s1", "s2", "s3"], name="split_0", axis=0),
        helper.make_node("Squeeze", ["s2", "squeeze_axes"], ["sq"], name="squeeze_0"),
        helper.make_node("Unsqueeze", ["sq", "unsqueeze_axes"], ["y"], name="unsqueeze_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "shape_concat_reshape_transpose_split_unsqueeze",
        [x],
        [y],
        initializer=[
            starts0, ends0, starts1, ends1, axes0, steps0,
            one, four, split_sizes, squeeze_axes, unsqueeze_axes,
        ],
    )
    return ensure_model(tmpdir, graph)


def test_shape_concat_reshape_transpose_slice_runs_on_ggml(
        suite_tmpdir, ep_library: Path) -> None:
    model_path = build_shape_concat_reshape_then_slice_chain_model(suite_tmpdir)
    rng = np.random.default_rng(43)
    inputs = {"x": rng.standard_normal((2, 3)).astype(np.float32)}
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    expected = cpu.run(["y"], inputs)[0]
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    # Slice sees a Transpose output whose input Reshape's shape ORT left
    # symbolic. Shape propagation must bridge through both Reshape and
    # Transpose for the outer Slice to accept.
    profile = end_profiling_profile(ggml)
    node_events = [e for e in profile if e.get("cat") == "Node"]
    providers = {e.get("args", {}).get("provider") for e in node_events}
    cpu_slices = [
        e for e in node_events
        if e.get("args", {}).get("provider") == "CPUExecutionProvider"
        and e.get("args", {}).get("op_name") == "Slice"
    ]
    assert not cpu_slices, f"Slice fell back to CPU: {cpu_slices}"
    assert "GGMLExecutionProvider" in providers, "no GGML partition executed"


def test_shape_concat_reshape_then_expand_runs_on_ggml(
        suite_tmpdir, ep_library: Path) -> None:
    model_path = build_shape_concat_reshape_then_expand_model(suite_tmpdir)
    rng = np.random.default_rng(53)
    inputs = {"x": rng.standard_normal((2, 3)).astype(np.float32)}
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    expected = cpu.run(["y"], inputs)[0]
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    profile = end_profiling_profile(ggml)
    node_events = [e for e in profile if e.get("cat") == "Node"]
    providers = {e.get("args", {}).get("provider") for e in node_events}
    cpu_expands = [
        e for e in node_events
        if e.get("args", {}).get("provider") == "CPUExecutionProvider"
        and e.get("args", {}).get("op_name") == "Expand"
    ]
    assert not cpu_expands, f"Expand fell back to CPU: {cpu_expands}"
    assert "GGMLExecutionProvider" in providers, "no GGML partition executed"


def test_shape_concat_reshape_transpose_split_unsqueeze_runs_on_ggml(
        suite_tmpdir, ep_library: Path) -> None:
    model_path = build_shape_concat_reshape_transpose_split_unsqueeze_model(suite_tmpdir)
    rng = np.random.default_rng(61)
    inputs = {"x": rng.standard_normal((2, 12)).astype(np.float32)}
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    expected = cpu.run(["y"], inputs)[0]
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    profile = end_profiling_profile(ggml)
    node_events = [e for e in profile if e.get("cat") == "Node"]
    providers = {e.get("args", {}).get("provider") for e in node_events}
    cpu_ops = [
        e for e in node_events
        if e.get("args", {}).get("provider") == "CPUExecutionProvider"
        and e.get("args", {}).get("op_name") in {"Split", "Squeeze", "Unsqueeze"}
    ]
    assert not cpu_ops, f"Split/Squeeze/Unsqueeze fell back to CPU: {cpu_ops}"
    assert "GGMLExecutionProvider" in providers, "no GGML partition executed"
