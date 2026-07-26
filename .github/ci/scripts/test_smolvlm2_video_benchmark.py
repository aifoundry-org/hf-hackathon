from __future__ import annotations

import os
import subprocess
import sys
import types
import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

if "fcntl" not in sys.modules:
    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_EX = 2
    fcntl.LOCK_UN = 8
    fcntl.flock = lambda *args, **kwargs: None
    sys.modules["fcntl"] = fcntl

import run_smolvlm2_video_benchmark as benchmark
import run_llama_server_benchmark as llama_benchmark


class SmolVLM2VideoBenchmarkTests(unittest.TestCase):
    def test_runtime_library_path_prefers_compatible_et_sdk_lib(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "lib").mkdir()
            with mock.patch.dict(
                os.environ,
                {
                    "ET_PLATFORM": str(root),
                    "ET_INSTALL": "/incompatible/et-install",
                    "ET_LIB_PATH": "/incompatible/host:/compatible/lib",
                    "LLAMA_CPP_ET_LD_LIBRARY_PATH": "",
                },
                clear=False,
            ):
                self.assertEqual(
                    benchmark.compatible_runtime_library_path(),
                    str(root / "lib"),
                )
                self.assertEqual(
                    llama_benchmark.compatible_runtime_library_path(),
                    str(root / "lib"),
                )

    def test_llama_revision_does_not_fall_through_to_parent_repository(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subprocess.run(
                ["git", "init", "-q", str(root)],
                check=True,
            )
            source = root / "vendored-source-without-git-metadata"
            source.mkdir()
            self.assertEqual(llama_benchmark.git_source_revision(source), "")

    def test_normalize_answer(self) -> None:
        self.assertEqual(benchmark.normalize_answer(" Giraffe.\n"), "giraffe")

    def test_extracts_and_restricts_fallback_ops(self) -> None:
        log = "\n".join(
            [
                "clip_ctx: CLIP using ET backend",
                "load_tensors: offloaded 33/33 layers to GPU",
                "warmup:           IM2COL: type = f32, ne = [1 1 1 1]",
                "warmup:             NORM: type = f32, ne = [1 1 1 1]",
                "srv log_server_r: done request: POST /completion 127.0.0.1 200",
                "ggml_et_init: using device ET",
            ]
        )
        self.assertEqual(benchmark.unsupported_vision_ops(log), {"IM2COL", "NORM"})
        self.assertEqual(
            benchmark.log_failures(log, mode="board", request_count=1, allowed_ops={"IM2COL", "NORM"}),
            [],
        )
        failures = benchmark.log_failures(log, mode="board", request_count=1, allowed_ops={"NORM"})
        self.assertIn("board: unexpected vision fallback ops: IM2COL", failures)

        strict_log = log + "\nwarmup: WARNING: the CLIP graph uses unsupported operators by the backend\n"
        strict_failures = benchmark.log_failures(strict_log, mode="board", request_count=1, allowed_ops=set())
        self.assertIn("board: vision graph contains CPU fallback operations", strict_failures)

    def test_correctness_requires_host_agreement_and_image_order(self) -> None:
        cases = [
            ({"name": "forward", "accepted_answers": ["giraffe"]}, [Path("cat"), Path("giraffe")]),
            ({"name": "reverse", "accepted_answers": ["cat"]}, [Path("giraffe"), Path("cat")]),
        ]
        host = [
            {"case": "forward", "kind": "correctness", "normalized_answer": "giraffe"},
            {"case": "reverse", "kind": "correctness", "normalized_answer": "cat"},
        ]
        board = [dict(item) for item in host]
        self.assertEqual(benchmark.correctness_failures(cases, host, board, ["forward", "reverse"]), [])

        board[1]["normalized_answer"] = "giraffe"
        failures = benchmark.correctness_failures(cases, host, board, ["forward", "reverse"])
        self.assertTrue(any("board answer" in failure for failure in failures))
        self.assertTrue(any("image-order check failed" in failure for failure in failures))

    def test_first_run_perplexity_ceiling(self) -> None:
        contract = {
            "first_run_perplexity": 22.2825,
            "max_relative_regression": 0.2,
            "maximum_perplexity": 26.739,
        }
        self.assertEqual(benchmark.maximum_perplexity(contract), 26.739)
        contract["maximum_perplexity"] = 26.7
        with self.assertRaisesRegex(ValueError, "internally inconsistent"):
            benchmark.maximum_perplexity(contract)

    def test_model_identity_requires_language_and_vision_structure(self) -> None:
        contract = {
            "architecture": {
                "language": {
                    "general_architecture": "llama",
                    "model_name": "SmolVLM2 500M Video Instruct",
                    "parameter_count_millions": 409.25,
                    "tensor_count": 291,
                    "block_count": 32,
                    "embedding_length": 960,
                    "feed_forward_length": 2560,
                    "attention_heads": 15,
                    "attention_kv_heads": 5,
                    "vocabulary_size": 49280,
                },
                "vision": {
                    "projector": "idefics3",
                    "tensor_count": 198,
                    "embedding_length": 768,
                    "attention_heads": 12,
                    "feed_forward_length": 3072,
                    "block_count": 12,
                    "projection_dimension": 960,
                    "image_size": 512,
                    "patch_size": 16,
                },
            }
        }
        log = """
general.architecture str = llama
general.name str = SmolVLM2 500M Video Instruct
model params = 409.25 M
loaded meta data with 75 key-value pairs and 291 tensors
llama.block_count u32 = 32
llama.embedding_length u32 = 960
llama.feed_forward_length u32 = 2560
llama.attention.head_count u32 = 15
llama.attention.head_count_kv u32 = 5
llama.vocab_size u32 = 49280
clip_model_loader: model name: SmolVLM2 500M Video Instruct
clip_model_loader: n_tensors: 198
load_hparams: projector: idefics3
load_hparams: n_embd: 768
load_hparams: n_head: 12
load_hparams: n_ff: 3072
load_hparams: n_layer: 12
load_hparams: projection_dim: 960
load_hparams: image_size: 512
load_hparams: patch_size: 16
"""
        self.assertEqual(benchmark.model_identity_failures(log, contract), [])
        self.assertIn(
            "loader log did not confirm vision block count",
            benchmark.model_identity_failures(
                log.replace("n_layer: 12", "n_layer: 6"), contract
            ),
        )

    def test_model_identity_allows_missing_kv_metadata_when_opted_out(self) -> None:
        contract = {
            "architecture": {
                "language": {
                    "general_architecture": "llama",
                    "model_name": "SmolVLM2 2.2B Instruct",
                    "parameter_count": {"value": 1.81, "unit": "B"},
                    "require_attention_kv_heads_in_metadata": False,
                    "tensor_count": 219,
                    "block_count": 24,
                    "embedding_length": 2048,
                    "feed_forward_length": 8192,
                    "attention_heads": 32,
                    "attention_kv_heads": 32,
                    "vocabulary_size": 49280,
                },
                "vision": {
                    "model_name": "SmolVLM2 2.2B Instruct",
                    "projector": "idefics3",
                    "tensor_count": 438,
                    "embedding_length": 1152,
                    "attention_heads": 16,
                    "feed_forward_length": 4304,
                    "block_count": 27,
                    "projection_dimension": 2048,
                    "image_size": 384,
                    "patch_size": 14,
                },
            }
        }
        log = """
general.architecture str = llama
general.name str = SmolVLM2 2.2B Instruct
model params = 1.81 B
loaded meta data with 75 key-value pairs and 219 tensors
llama.block_count u32 = 24
llama.embedding_length u32 = 2048
llama.feed_forward_length u32 = 8192
llama.attention.head_count u32 = 32
llama.vocab_size u32 = 49280
clip_model_loader: model name: SmolVLM2 2.2B Instruct
clip_model_loader: n_tensors: 438
load_hparams: projector: idefics3
load_hparams: n_embd: 1152
load_hparams: n_head: 16
load_hparams: n_ff: 4304
load_hparams: n_layer: 27
load_hparams: projection_dim: 2048
load_hparams: image_size: 384
load_hparams: patch_size: 14
"""
        self.assertEqual(benchmark.model_identity_failures(log, contract), [])

    def test_model_identity_qwen3vl_real_loader_format(self) -> None:
        contract = {
            "architecture": {
                "language": {
                    "general_architecture": "qwen3vl",
                    "metadata_key_prefix": "qwen3vl",
                    "parameter_count": {"value": 1.72, "unit": "B"},
                    "require_model_name": False,
                    "vocabulary": {"source": "n_vocab", "size": 151936},
                    "tensor_count": 310,
                    "block_count": 28,
                    "embedding_length": 2048,
                    "feed_forward_length": 6144,
                    "attention_heads": 16,
                    "attention_kv_heads": 8,
                },
                "vision": {
                    "require_model_name": False,
                    "projector": "qwen3vl_merger",
                    "tensor_count": 316,
                    "embedding_length": 1024,
                    "attention_heads": 16,
                    "feed_forward_length": 4096,
                    "block_count": 24,
                    "projection_dimension": 2048,
                    "image_size": 768,
                    "patch_size": 16,
                },
            }
        }
        log = """
general.architecture str = qwen3vl
model params = 1.72 B
loaded meta data with 40 key-value pairs and 310 tensors
qwen3vl.block_count u32 = 28
qwen3vl.embedding_length u32 = 2048
qwen3vl.feed_forward_length u32 = 6144
qwen3vl.attention.head_count u32 = 16
qwen3vl.attention.head_count_kv u32 = 8
n_vocab = 151936
clip_model_loader: n_tensors: 316
load_hparams: projector: qwen3vl_merger
load_hparams: n_embd: 1024
load_hparams: n_head: 16
load_hparams: n_ff: 4096
load_hparams: n_layer: 24
load_hparams: projection_dim: 2048
load_hparams: image_size: 768
load_hparams: patch_size: 16
"""
        self.assertEqual(benchmark.model_identity_failures(log, contract), [])
        # Unit B must match the B loader print; do not accept a converted M line.
        self.assertIn(
            "loader log did not confirm language parameter count",
            benchmark.model_identity_failures(
                log.replace("model params = 1.72 B", "model params = 1720.00 M"),
                contract,
            ),
        )

    def test_model_identity_qwen3vl_negative_cases(self) -> None:
        contract = {
            "architecture": {
                "language": {
                    "general_architecture": "qwen3vl",
                    "metadata_key_prefix": "qwen3vl",
                    "parameter_count": {"value": 1.72, "unit": "B"},
                    "require_model_name": False,
                    "vocabulary": {"source": "n_vocab", "size": 151936},
                    "tensor_count": 310,
                    "block_count": 28,
                    "embedding_length": 2048,
                    "feed_forward_length": 6144,
                    "attention_heads": 16,
                    "attention_kv_heads": 8,
                },
                "vision": {
                    "require_model_name": False,
                    "projector": "qwen3vl_merger",
                    "tensor_count": 316,
                    "embedding_length": 1024,
                    "attention_heads": 16,
                    "feed_forward_length": 4096,
                    "block_count": 24,
                    "projection_dimension": 2048,
                    "image_size": 768,
                    "patch_size": 16,
                },
            }
        }
        log = """
general.architecture str = qwen3vl
model params = 1.72 B
loaded meta data with 40 key-value pairs and 310 tensors
qwen3vl.block_count u32 = 28
qwen3vl.embedding_length u32 = 2048
qwen3vl.feed_forward_length u32 = 6144
qwen3vl.attention.head_count u32 = 16
qwen3vl.attention.head_count_kv u32 = 8
n_vocab = 151936
clip_model_loader: n_tensors: 316
load_hparams: projector: qwen3vl_merger
load_hparams: n_embd: 1024
load_hparams: n_head: 16
load_hparams: n_ff: 4096
load_hparams: n_layer: 24
load_hparams: projection_dim: 2048
load_hparams: image_size: 768
load_hparams: patch_size: 16
"""
        self.assertIn(
            "loader log did not confirm language block count",
            benchmark.model_identity_failures(
                log.replace("qwen3vl.block_count u32 = 28", "llama.block_count u32 = 28"),
                contract,
            ),
        )
        self.assertIn(
            "loader log did not confirm language parameter count",
            benchmark.model_identity_failures(
                log.replace("model params = 1.72 B", "model params = 1.80 B"),
                contract,
            ),
        )
        self.assertIn(
            "loader log did not confirm language vocabulary",
            benchmark.model_identity_failures(
                log.replace("n_vocab = 151936", "n_vocab = 151000"),
                contract,
            ),
        )
        self.assertIn(
            "loader log did not confirm language tensor count",
            benchmark.model_identity_failures(
                log.replace("and 310 tensors", "and 300 tensors"),
                contract,
            ),
        )
        self.assertIn(
            "loader log did not confirm vision projector",
            benchmark.model_identity_failures(
                log.replace("projector: qwen3vl_merger", "projector: idefics3"),
                contract,
            ),
        )

    def test_model_identity_reports_incomplete_contract(self) -> None:
        contract = {
            "architecture": {
                "language": {
                    "general_architecture": "qwen3vl",
                    "tensor_count": 310,
                    "block_count": 28,
                    "embedding_length": 2048,
                    "feed_forward_length": 6144,
                    "attention_heads": 16,
                    "attention_kv_heads": 8,
                },
                "vision": {
                    "projector": "qwen3vl_merger",
                    "tensor_count": 316,
                    "embedding_length": 1024,
                    "attention_heads": 16,
                    "feed_forward_length": 4096,
                    "block_count": 24,
                    "projection_dimension": 2048,
                    "image_size": 768,
                    "patch_size": 16,
                },
            }
        }
        failures = benchmark.model_identity_failures("", contract)
        self.assertTrue(
            any(
                failure.startswith(
                    "identity contract missing language.metadata_key_prefix"
                )
                for failure in failures
            ),
        )
        self.assertIn(
            "identity contract missing language.parameter_count or parameter_count_millions",
            failures,
        )
        self.assertIn(
            "identity contract missing language.vocabulary or vocabulary_size",
            failures,
        )
        self.assertIn(
            "identity contract missing language.model_name or language.require_model_name",
            failures,
        )
        self.assertIn(
            "identity contract missing vision.model_name or vision.require_model_name",
            failures,
        )

    def test_artifacts_must_match_frozen_hash_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "model.gguf"
            path.write_bytes(b"frozen model")
            baseline = {"sha256": benchmark.sha256_file(path), "size_bytes": path.stat().st_size}
            self.assertIsNone(benchmark.artifact_policy_error(path, baseline, kind="model"))
            path.write_bytes(b"different model")
            self.assertIn(
                "frozen baseline",
                benchmark.artifact_policy_error(path, baseline, kind="model") or "",
            )

    def test_selects_only_fast_ci_correctness_cases(self) -> None:
        cases = [
            ({"name": "video_dog"}, [Path("dog")]),
            ({"name": "video_bridge"}, [Path("bridge")]),
        ]
        self.assertEqual(
            benchmark.selected_correctness_cases(cases, {"ci_cases": ["video_dog"]}),
            [cases[0]],
        )
        with self.assertRaisesRegex(ValueError, "not in the input contract"):
            benchmark.selected_correctness_cases(cases, {"ci_cases": ["missing"]})

    def test_firmware_cycles_are_scoped_to_request_timestamps(self) -> None:
        def extra(key: str, index: int, data: object) -> dict[str, object]:
            return {"key": key, "value": {"index": index, "data": data}}

        events = {
            "value0": {
                "class": "StartProfiling",
                "timeStamp": {"time_since_epoch": {"count": 10}},
                "extra": [extra("version", 8, 3)],
            },
            "value1": {
                "class": "KernelLaunch",
                "timeStamp": {"time_since_epoch": {"count": 100}},
                "extra": [extra("event", 1, 41), extra("kernel_id", 4, 7)],
            },
            "value2": {
                "class": "ResponseReceived",
                "timeStamp": {"time_since_epoch": {"count": 120}},
                "extra": [
                    extra("rsp_type", 5, 2),
                    extra("device_cmd_exec_dur", 0, 1234),
                    extra("device_cmd_start_ts", 0, 9000),
                    extra("device_cmd_wait_dur", 0, 5),
                    extra("event", 1, 41),
                ],
            },
            "value3": {
                "class": "KernelLaunch",
                "timeStamp": {"time_since_epoch": {"count": 300}},
                "extra": [extra("event", 1, 42), extra("kernel_id", 4, 7)],
            },
            "value4": {
                "class": "ResponseReceived",
                "timeStamp": {"time_since_epoch": {"count": 320}},
                "extra": [
                    extra("rsp_type", 5, 2),
                    extra("device_cmd_exec_dur", 0, 9999),
                    extra("device_cmd_start_ts", 0, 10000),
                    extra("device_cmd_wait_dur", 0, 4),
                    extra("event", 1, 42),
                ],
            },
            "value5": {
                "class": "EndProfiling",
                "timeStamp": {"time_since_epoch": {"count": 400}},
                "extra": [],
            },
        }
        performance = {
            "profile_schema_version": 3,
            "profile_class": "ResponseReceived",
            "primary_counter": "device_cmd_exec_dur",
            "minimum_kernel_launches_per_request": 1,
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            profile = root / "trace.json"
            kernel_map = root / "kernels.json"
            profile.write_text(json.dumps(events))
            kernel_map.write_text(json.dumps({"vision_fused": 7}))
            measurements, summary = benchmark.read_firmware_cycles(
                profile,
                kernel_map,
                [{"request_id": "measured", "started_ns": 90, "ended_ns": 200}],
                performance,
            )
        self.assertEqual(measurements["measured"]["cycles"], 1234)
        self.assertEqual(measurements["measured"]["kernel_launches"], 1)
        self.assertEqual(measurements["measured"]["kernels"], ["vision_fused"])
        self.assertEqual(summary["matched_kernel_responses"], 1)


if __name__ == "__main__":
    unittest.main()
