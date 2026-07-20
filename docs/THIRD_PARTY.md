# Third-Party Components

The first-party contents in this repository are licensed under **Apache-2.0**
(see [`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE)). This includes the
vendored Esperanto sys-emu firmware under
`.github/ci/firmware/esperanto-fw/`. The components below are external
components and retain their own upstream terms. This file is the top-level
inventory; per-port `THIRD_PARTY.md` files hold the detailed records.

| Component | Path | Type | License | Notes |
|-----------|------|------|---------|-------|
| ET `llama.cpp` fork | `ported_models/llama_cpp_et/src/llama.cpp-et` | git submodule (pointer) | MIT | Upstream `aifoundry-org/llama.cpp`, branch `et`. License travels with the submodule. See `ported_models/llama_cpp_et/THIRD_PARTY.md`. |
| GGONNX | `ported_models/ggonnx/src/ggonnx` | vendored source | Pending (expected Apache-2.0) | Upstream `marty1885/ggonnx` had no LICENSE at vendoring time; license grant being secured with the author. See `ported_models/ggonnx/THIRD_PARTY.md`. |
| YOLOv10n pinned ONNX | `local-artifacts/yolov10n_hf_reference/model.onnx` (not committed) | downloaded model graph and weights | AGPL-3.0 | `onnx-community/yolov10n` revision `57657320425ee34056408a57ad9d29c4d4815bd8`; see `ported_models/yolov10n_hf_reference/THIRD_PARTY.md`. |
| Model weights (GGUF, ONNX) | not committed | downloaded at runtime | Per upstream model card | Fetched on the board host from Hugging Face / source URLs declared in each port's `artifacts.json`; each model retains its own license. |

## How licensing is structured here

- `LICENSE` (Apache-2.0) applies to this repository's **own** contents: the CI
  scripts, configs, docs, porting harness, and vendored Esperanto sys-emu
  firmware.
- `NOTICE` carries the Apache attribution required for redistribution.
- Bundled third-party source/binaries keep their upstream terms; do not assume
  Apache-2.0 covers anything listed above.
- Each `ported_models/<port>/THIRD_PARTY.md` records the upstream URL, pinned
  revision, vendoring date, and license status for that port.

If you add a port that vendors or submodules external code, record it here and in
your port's `THIRD_PARTY.md` (see [`SUBMISSION_GUIDE.md`](SUBMISSION_GUIDE.md)).
