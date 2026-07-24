#!/usr/bin/env python3
"""
Run the fused FFN kernel on ET-SoC1.

Uses the pre-built erbium_soc1sim_argbuf_dynmem launcher at /opt/et/bin/.

Usage:
  ./run_kernel.py --weights model_q8_0.bin --input act.bin --output result.bin

The weights file must contain three Q8_0 matrices contiguously:
  [W_gate: inter x hidden] [W_up: inter x hidden] [W_down: hidden x inter]
  Each matrix is row-major Q8_0, each row = ceil(dim/32) blocks of 34 bytes.

The input file contains [hidden] float32 values.
"""

import argparse
import struct
import subprocess
import sys
import os
import tempfile

Q8_BYTES = 34  # sizeof(block_q8_0)

def make_params_bin(hidden, inter):
    """Build the FFNParams binary blob."""
    H = hidden
    I = inter
    H_blk = H // 32
    I_blk = I // 32

    # Layout within the device buffer
    param_off    = 0
    param_sz     = 80  # sizeof(FFNParams) = 10 * 8
    input_off    = 1024
    input_sz     = H * 4
    wgate_off    = (input_off + input_sz + 4095) & ~4095
    wgate_sz     = I * H_blk * Q8_BYTES
    wup_off      = wgate_off + wgate_sz
    wup_sz       = I * H_blk * Q8_BYTES
    wdown_off    = wup_off + wup_sz
    wdown_sz     = H * I_blk * Q8_BYTES
    scratch_off  = (wdown_off + wdown_sz + 4095) & ~4095
    scratch_sz   = I * 3 * 4
    output_off   = (scratch_off + scratch_sz + 4095) & ~4095
    output_sz    = H * 4

    buf_size = output_off + output_sz + 4096

    params = struct.pack(
        '<10q',  # 10 x signed 64-bit little-endian
        input_off,
        wgate_off,
        wup_off,
        wdown_off,
        output_off,
        scratch_off,
        H,
        I,
        H_blk,
        I_blk,
    )

    return params, {
        'input_off': input_off,
        'wgate_off': wgate_off,
        'wup_off': wup_off,
        'wdown_off': wdown_off,
        'output_off': output_off,
        'scratch_off': scratch_off,
        'param_off': param_off,
        'buf_size': buf_size,
    }

def main():
    parser = argparse.ArgumentParser(description='Run fused FFN kernel on ET-SoC1')
    parser.add_argument('--elf', default=None,
                        help='Path to kernel .elf (default: build/llama32_fused_ffn.elf)')
    parser.add_argument('--weights', required=True,
                        help='Path to Q8_0 weight file (gate + up + down concatenated)')
    parser.add_argument('--input', required=True,
                        help='Path to input activation (.bin of float32)')
    parser.add_argument('--output', required=True,
                        help='Path to write output (.bin of float32)')
    parser.add_argument('--hidden', type=int, default=2048,
                        help='Hidden dimension (default: 2048)')
    parser.add_argument('--inter', type=int, default=8192,
                        help='Intermediate dimension (default: 8192)')
    parser.add_argument('--launcher', default='/opt/et/bin/erbium_soc1sim_argbuf_dynmem',
                        help='Launcher binary path')
    parser.add_argument('--shire', type=int, default=0,
                        help='Shire to run on (default: 0)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout in seconds (default: 30)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print debug info')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    elf_path = args.elf or os.path.join(script_dir, 'build', 'llama32_fused_ffn.elf')

    if not os.path.exists(elf_path):
        print(f"Error: kernel ELF not found at {elf_path}")
        print("Build it first: cd ported_models/llama_cpp_et/kernels && bash build.sh")
        sys.exit(1)

    if not os.path.exists(args.weights):
        print(f"Error: weights file not found: {args.weights}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    # Build param blob
    params, layout = make_params_bin(args.hidden, args.inter)
    buf_size = layout['buf_size']

    # Create temporary dir for data files
    with tempfile.TemporaryDirectory() as tmpdir:
        param_path = os.path.join(tmpdir, 'params.bin')
        with open(param_path, 'wb') as f:
            f.write(params)

        # Build launcher command
        cmd = [
            args.launcher,
            '--elf-load', elf_path,
            '--device', 'soc1sim',
            '--shire', str(args.shire),
            '--timeout', str(args.timeout),
            '--mem_size', str(buf_size),
            '--file_load', f"0x{layout['param_off']:x},{param_path}",
            '--file_load', f"0x{layout['input_off']:x},{args.input}",
            '--file_load', f"0x{layout['wgate_off']:x},{args.weights}",
            '--dump_after', f"{tmpdir}/output.bin",
        ]

        if args.verbose:
            print(f"Buffer: {buf_size} bytes")
            print(f"  params   @ 0x{layout['param_off']:x}")
            print(f"  input    @ 0x{layout['input_off']:x}")
            print(f"  W_gate   @ 0x{layout['wgate_off']:x}")
            print(f"  W_up     @ 0x{layout['wup_off']:x}")
            print(f"  W_down   @ 0x{layout['wdown_off']:x}")
            print(f"  scratch  @ 0x{layout['scratch_off']:x}")
            print(f"  output   @ 0x{layout['output_off']:x}")
            print(f"Running: {' '.join(cmd)}")

        # Run
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Kernel failed (exit {result.returncode})")
            print(result.stderr)
            print(result.stdout)
            sys.exit(1)

        if args.verbose:
            print(result.stdout)

        # Read back output
        output_path = os.path.join(tmpdir, 'output.bin')
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                data = f.read()
            with open(args.output, 'wb') as f:
                f.write(data)

            # Print first few values
            vals = struct.unpack(f'<{args.hidden}f', data[:args.hidden*4])
            print(f"Output[0..4]: {' '.join(f'{v:.6f}' for v in vals[:5])}")
            print(f"Output written to {args.output}")

            # Extract timing from launcher output
            for line in result.stdout.split('\n'):
                if 'Kernel wait seconds' in line or 'completed' in line.lower():
                    print(line)
        else:
            print("No output file produced")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)

if __name__ == '__main__':
    main()
