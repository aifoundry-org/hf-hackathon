# dncnn20l64 quality report

- Model: `deepinv/dncnn` (sigma=2 grayscale), runtime deepinv==0.4.1 / torch (CPU).
- Operating point: sigma=2 on [0,1] (as trained). Image 64x64.
- Oracle = torch FP32 baseline; host int8 = numpy debug proxy for the ET-SoC1 kernel.
- Noisy input PSNR vs clean: **41.92 dB**.

| result | PSNR vs clean | SSIM vs clean | denoise gain | max_abs vs oracle | rmse vs oracle |
|---|---|---|---|---|---|
| torch_oracle_fp32 | 51.03 dB | 0.9981 | +9.11 dB | - | - |
| host_int8_debug | 50.61 dB | 0.9979 | +8.69 dB | 1 | 0.341 |
| board_int8 | 50.59 dB | 0.9979 | +8.67 dB | 1 | 0.341 |

The board int8 row is measured from a real board dump via `--dump <board_output>`. The board kernel is deterministic integer arithmetic and matches the PyTorch oracle at max_abs=1 on the fixed gate input (board-verified, 3/3 runs). The uint8_npy gate is set to max_abs<=2 -- a 1-unit margin over the measured value.
