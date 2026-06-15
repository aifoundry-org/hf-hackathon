#!/usr/bin/env python3

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import onnxruntime as ort

_MOSAIC_MODEL_URL = (
    "https://github.com/onnx/models/raw/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx"
)

def model_cache_dir() -> Path:
    override = os.environ.get("GGONNX_TEST_MODEL_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "ggonnx-test" / "models"

def download_model(destination: Path):
    if destination.exists():
        return
    print(f"Downloading model to {destination}...", file=sys.stderr)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    def report(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 1e2 / total_size
            s = f"\rProgress: {percent:5.1f}% ({read_so_far}/{total_size} bytes)"
        else:
            s = f"\rProgress: {read_so_far} bytes"
        sys.stderr.write(s)
        sys.stderr.flush()

    urllib.request.urlretrieve(_MOSAIC_MODEL_URL, destination, reporthook=report)
    print("\nDownload complete.", file=sys.stderr)

_EP_REGISTERED = False

def register_ep(ep_library: Path):
    global _EP_REGISTERED
    if _EP_REGISTERED:
        return
    if not ep_library.exists():
        raise FileNotFoundError(f"EP library not found at {ep_library}. Please build the project first.")
    ort.register_execution_provider_library("GGONNX", str(ep_library))
    _EP_REGISTERED = True

def create_session(model_path: Path, provider: str, ep_library: Path = None, ggml_device: str = None, n_threads: int = None, matmul_precision: str = None):
    so = ort.SessionOptions()
    # Suppress warnings (0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal)
    so.log_severity_level = 3
    
    if provider == "onnx-cpu":
        so.add_provider("CPUExecutionProvider", {})
    elif provider == "ggonnx":
        register_ep(ep_library)
        if ggml_device:
            so.add_session_config_entry("ep.ggmlexecutionprovider.ggml_device_list", ggml_device)
        if n_threads:
            so.add_session_config_entry("ep.ggmlexecutionprovider.n_threads", str(n_threads))
        if matmul_precision:
            so.add_session_config_entry("ep.ggmlexecutionprovider.matmul_precision", matmul_precision)
        
        devices = [d for d in ort.get_ep_devices() if d.ep_name == "GGMLExecutionProvider"]
        if not devices:
            raise RuntimeError("GGMLExecutionProvider not found in ORT devices")
        so.add_provider_for_devices(devices, {})
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ort.InferenceSession(str(model_path), sess_options=so)

def benchmark(session, label, iterations=50, warmup=10):
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    
    # Handle symbolic dimensions (usually 'batch', etc.)
    concrete_shape = [d if isinstance(d, int) else 1 for d in input_shape]
    dummy_input = np.random.randn(*concrete_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        session.run(None, {input_name: dummy_input})
    end = time.perf_counter()
    
    elapsed = end - start
    avg_time = elapsed / iterations
    inf_per_sec = 1.0 / avg_time
    
    print(f"{label:25}: {inf_per_sec:10.2f} inf/sec")

def main():
    parser = argparse.ArgumentParser(description="Benchmark GGONNX vs ONNX CPU")
    parser.add_argument("--ep-library", type=Path, default="build/libggonnx_ep.so", help="Path to libggonnx_ep.so")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations for benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--threads", type=int, default=os.cpu_count() // 2, help="Number of threads for GGML CPU")
    args = parser.parse_args()

    model_path = model_cache_dir() / "mosaic-9.onnx"
    download_model(model_path)
    
    ep_library = args.ep_library.resolve()
    
    print(f"Benchmarking Mosaic model ({args.iterations} iterations, {args.warmup} warmup, {args.threads} threads)")
    print("-" * 50)
    
    # 1. ONNX CPU
    try:
        sess = create_session(model_path, "onnx-cpu")
        benchmark(sess, "1. ONNX CPU", args.iterations, args.warmup)
    except Exception as e:
        print(f"ONNX CPU benchmark failed: {e}")

    # 2. GGONNX CPU (forcing ggml-cpu)
    try:
        sess = create_session(model_path, "ggonnx", ep_library, ggml_device="CPU", n_threads=args.threads)
        benchmark(sess, f"2. GGONNX CPU ({args.threads} th)", args.iterations, args.warmup)
    except Exception as e:
        print(f"GGONNX CPU benchmark failed: {e}")
        # Diagnostic: what are the available GGML devices?
        print("Available GGML devices:")
        try:
            for d in ort.get_ep_devices():
                if d.ep_name == "GGMLExecutionProvider":
                    print(f"  - {d}")
        except:
            pass

    # 3. GGONNX Default
    try:
        # Note: GGONNX prints its device selection to stderr on session creation.
        # It will now default to half of available threads internally.
        sess = create_session(model_path, "ggonnx", ep_library, n_threads=args.threads)
        benchmark(sess, f"3. GGONNX Default ({args.threads} th)", args.iterations, args.warmup)
    except Exception as e:
        print(f"GGONNX Default benchmark failed: {e}")

if __name__ == "__main__":
    main()
