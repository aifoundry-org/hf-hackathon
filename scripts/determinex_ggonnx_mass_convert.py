#!/usr/bin/env python3
"""Determinex GGONNX Mass Converter Loop

This is the unattended lane for Pillar 4: Mass transpilation of ONNX vision models.
It uses robust locking and state management per Determinex corpus principles.
"""
import argparse
import datetime as _dt
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
STATE_PATH = LOG_DIR / "ggonnx_state.json"
WATCH_LOCK_PATH = LOG_DIR / "ggonnx_watch.lock"
PORTED_MODELS_DIR = ROOT / "ported_models"

# Found live 2026-07-10: the original 4 "onnx-models/*" repo IDs here were
# all fabricated -- none exist on HuggingFace (verified via
# HfApi.model_info(), all 404/401). Replaced with real, ungated repos from
# trusted orgs (onnx-community is HF's own maintained ONNX-export org; the
# bare "onnx" org is the official ONNX model zoo), each confirmed to
# actually contain a .onnx file before being added here. Dropped the
# yolov8n slot outright: no ONNX export exists in the official
# Ultralytics/YOLOv8 repo (only .pt weights), and no community re-export
# cleared the same trust bar as the other three -- forcing a low-confidence
# pick just to keep a 4th entry isn't worth it, and a YOLOv8-style detector
# is already well-covered by this repo's own yolo_m30 kernel.
MODELS = [
    {"name": "resnet18", "repo": "onnx-community/resnet-18-ONNX"},
    {"name": "mobilenetv2", "repo": "onnx-community/mobilenet_v2_1.0_224"},
    {"name": "efficientnet", "repo": "onnx/EfficientNet-Lite4"},
]

def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            try:
                code = wintypes.DWORD()
                if k32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return code.value == 259
                return True
            finally:
                k32.CloseHandle(h)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False
    except PermissionError:
        return True

def acquire_watch_lock(path: Path = WATCH_LOCK_PATH) -> tuple[bool, int | None, str]:
    """Acquire the singleton watch-loop lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "ts": _now()}
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
            return True, fd, ""
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8", errors="replace") or "{}")
            except Exception:
                existing = {}
            try:
                pid = int(existing.get("pid") or 0)
            except Exception:
                pid = 0
            if pid and not _pid_running(pid):
                try:
                    path.unlink()
                    continue
                except (FileNotFoundError, OSError):
                    pass
            return False, None, f"locked by pid {pid}"

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Determinex GGONNX Mass Converter")
    parser.add_argument("--watch", action="store_true", help="Run in continuous watch mode")
    parser.add_argument("--interval", type=int, default=300, help="Watch interval in seconds")
    parser.add_argument("--limit", type=int, default=10, help="Max models to process per run")
    parser.add_argument("--timeout", type=int, default=1800, help="Subprocess timeout in seconds")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Record planned work without invoking ggonnx or marking models done",
    )
    args = parser.parse_args()

    locked, fd, msg = acquire_watch_lock()
    if not locked:
        print(f"Bailing: {msg}")
        return 1

    try:
        while True:
            state = load_state()
            processed = 0

            for model in MODELS:
                if processed >= args.limit:
                    break
                
                name = model["name"]
                repo = model["repo"]

                if name in state and state[name].get("status") == "done":
                    continue

                print(f"[{_now()}] Processing {name} ({repo})...")
                
                try:
                    if args.dry_run:
                        state[name] = {"status": "planned", "ts": _now(), "repo": repo}
                        print(f"[{_now()}] Planned {name}; dry-run did not invoke ggonnx.")
                        save_state(state)
                        processed += 1
                        continue

                    out_dir = PORTED_MODELS_DIR / name
                    out_dir.mkdir(parents=True, exist_ok=True)

                    print(f"Running ggonnx on {repo}...")
                    subprocess.run(
                        ["ggonnx", repo, "--out", str(out_dir)],
                        check=True,
                        timeout=args.timeout,
                        capture_output=True,
                        text=True
                    )

                    state[name] = {"status": "done", "ts": _now(), "repo": repo}
                    print(f"[{_now()}] Successfully generated {name} artifacts.")
                except FileNotFoundError:
                    print(f"[{_now()}] FAILED {name}: ggonnx binary not found")
                    state[name] = {
                        "status": "error",
                        "ts": _now(),
                        "repo": repo,
                        "error": "ggonnx binary not found",
                    }
                except subprocess.TimeoutExpired:
                    print(f"[{_now()}] TIMEOUT on {name} after {args.timeout}s.")
                    state[name] = {"status": "timeout", "ts": _now()}
                except subprocess.CalledProcessError as e:
                    print(f"[{_now()}] FAILED {name}: {e.stderr}")
                    state[name] = {"status": "error", "ts": _now(), "error": e.stderr}
                except Exception as e:
                    print(f"[{_now()}] UNEXPECTED ERROR {name}: {e}")
                    state[name] = {"status": "error", "ts": _now(), "error": str(e)}

                save_state(state)
                processed += 1

            if not args.watch:
                break
            
            print(f"[{_now()}] Pass complete. Sleeping {args.interval}s...")
            time.sleep(args.interval)

    finally:
        if fd is not None:
            os.close(fd)
            try:
                WATCH_LOCK_PATH.unlink()
            except OSError:
                pass

if __name__ == "__main__":
    import sys
    sys.exit(main())
