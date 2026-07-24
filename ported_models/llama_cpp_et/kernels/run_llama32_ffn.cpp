/*
 * Host launcher for the fused FFN kernel on ET-SoC1.
 *
 * Build:
 *   g++ -std=c++17 -O2 \
 *     -I/opt/et/include \
 *     -L/opt/et/lib \
 *     -o run_llama32_ffn run_llama32_ffn.cpp \
 *     -lETRuntime -lETDeviceLayer -lpthread
 *
 * Run:
 *   ./run_llama32_ffn \
 *     --elf ported_models/llama_cpp_et/kernels/build/llama32_fused_ffn.elf \
 *     --weights /data/models/llama32_1b/llama-3.2-1b-q8_0.bin \
 *     --input /tmp/input_act.bin \
 *     --output /tmp/output.bin \
 *     --hidden 2048 --inter 8192 \
 *     --shires 32 --timeout 30
 *
 * The weights file must contain the Q8_0 weight matrices in GGML row-major
 * order (rows × blocks × 34 bytes each, where blocks = hidden/32 or inter/32).
 *
 * The input file must contain M×hidden f32 values (M=1 for token generation).
 *
 * The output file will contain hidden f32 values.
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <device-layer/IDeviceLayer.h>
#include <hostUtils/logging/Logger.h>
#include <runtime/IRuntime.h>
#include <runtime/Types.h>

namespace fs = std::filesystem;

/* ----------------------------------------------------------------- */
/* Kernel parameter block — must match the kernel's struct ffn_params */
/* ----------------------------------------------------------------- */
#pragma pack(push, 1)
struct FFNParams {
    uint64_t input_off;      /* offset of input [hidden] f32 */
    uint64_t wgate_off;      /* offset of W_gate [inter, hidden] Q8_0 */
    uint64_t wup_off;        /* offset of W_up   [inter, hidden] Q8_0 */
    uint64_t wdown_off;      /* offset of W_down [hidden, inter] Q8_0 */
    uint64_t output_off;     /* offset of output [hidden] f32 */
    uint64_t scratch_off;    /* offset of scratch [inter×3] f32 */
    int64_t  hidden;         /* H — 2048 */
    int64_t  inter;          /* I — 8192 */
    int64_t  h_blocks;       /* H / 32 */
    int64_t  i_blocks;       /* I / 32 */
};
#pragma pack(pop)

#define Q8_BYTES 34  /* sizeof(block_q8_0) */

/* ----------------------------------------------------------------- */
/* Options                                                           */
/* ----------------------------------------------------------------- */
struct Options {
    std::string elf_path;
    std::string weights_path;
    std::string input_path;
    std::string output_path;
    int64_t hidden = 2048;
    int64_t inter = 8192;
    uint32_t shires = 32;
    uint64_t timeout_secs = 30;
    bool verbose = false;
};

Options parse_args(int argc, char** argv) {
    static const struct option long_opts[] = {
        {"elf",     required_argument, nullptr, 'e'},
        {"weights", required_argument, nullptr, 'w'},
        {"input",   required_argument, nullptr, 'i'},
        {"output",  required_argument, nullptr, 'o'},
        {"hidden",  required_argument, nullptr, 'n'},
        {"inter",   required_argument, nullptr, 'r'},
        {"shires",  required_argument, nullptr, 's'},
        {"timeout", required_argument, nullptr, 't'},
        {"verbose", no_argument,       nullptr, 'v'},
        {"help",    no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    Options opts;
    int c;
    while ((c = getopt_long(argc, argv, "vh", long_opts, nullptr)) != -1) {
        switch (c) {
        case 'e': opts.elf_path = optarg; break;
        case 'w': opts.weights_path = optarg; break;
        case 'i': opts.input_path = optarg; break;
        case 'o': opts.output_path = optarg; break;
        case 'n': opts.hidden = std::stoll(optarg); break;
        case 'r': opts.inter = std::stoll(optarg); break;
        case 's': opts.shires = std::stoul(optarg); break;
        case 't': opts.timeout_secs = std::stoull(optarg); break;
        case 'v': opts.verbose = true; break;
        case 'h':
            std::cerr << "Usage: " << argv[0] << " --elf <kernel.elf> --weights <q8_0.bin> --input <act.bin> --output <out.bin>\n"
                      << "  [--hidden N] [--inter N] [--shires N] [--timeout N] [--verbose]\n";
            std::exit(0);
        default:
            std::cerr << "Try --help\n";
            std::exit(1);
        }
    }

    if (opts.elf_path.empty() || opts.weights_path.empty() || opts.input_path.empty() || opts.output_path.empty()) {
        std::cerr << "Error: --elf, --weights, --input, and --output are required\n";
        std::exit(1);
    }

    return opts;
}

std::vector<char> read_file(const std::string& path) {
    auto size = fs::file_size(path);
    std::vector<char> buf(size);
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Error: cannot open " << path << "\n"; std::exit(1); }
    f.read(buf.data(), size);
    return buf;
}

void write_file(const std::string& path, const char* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Error: cannot write " << path << "\n"; std::exit(1); }
    f.write(data, size);
}

/* ----------------------------------------------------------------- */
/* Main                                                              */
/* ----------------------------------------------------------------- */
int main(int argc, char** argv) {
    auto opts = parse_args(argc, argv);

    int64_t H = opts.hidden;
    int64_t I = opts.inter;
    int64_t H_blk = H / 32;  /* Q8_0 blocks per row of a hidden-dim weight */
    int64_t I_blk = I / 32;  /* Q8_0 blocks per row of an inter-dim weight */

    /* Buffer layout */
    uint64_t param_off    = 0;
    size_t   param_sz     = sizeof(FFNParams);
    uint64_t input_off    = 1024;                        /* 1KB aligned */
    size_t   input_sz     = H * sizeof(float);           /* [H] f32 */
    uint64_t wgate_off    = input_off + input_sz + 4096; /* page align */
    size_t   wgate_sz     = I * H_blk * Q8_BYTES;        /* [I, H_blk, 34] */
    uint64_t wup_off      = wgate_off + wgate_sz;
    size_t   wup_sz       = I * H_blk * Q8_BYTES;        /* [I, H_blk, 34] */
    uint64_t wdown_off    = wup_off + wup_sz;
    size_t   wdown_sz     = H * I_blk * Q8_BYTES;        /* [H, I_blk, 34] */
    uint64_t scratch_off  = wdown_off + wdown_sz + 4096;
    size_t   scratch_sz   = I * 3 * sizeof(float);       /* gate + up + gated */
    uint64_t output_off   = scratch_off + scratch_sz + 4096;
    size_t   output_sz    = H * sizeof(float);

    uint64_t buf_size = output_off + output_sz + 4096;   /* total buffer */

    if (opts.verbose) {
        std::cerr << "Buffer layout:\n"
                  << "  params:   0x" << std::hex << param_off << " (" << param_sz << ")\n"
                  << "  input:    0x" << input_off << " (" << input_sz << ")\n"
                  << "  W_gate:   0x" << wgate_off << " (" << wgate_sz << ")\n"
                  << "  W_up:     0x" << wup_off << " (" << wup_sz << ")\n"
                  << "  W_down:   0x" << wdown_off << " (" << wdown_sz << ")\n"
                  << "  scratch:  0x" << scratch_off << " (" << scratch_sz << ")\n"
                  << "  output:   0x" << output_off << " (" << output_sz << ")\n"
                  << "  total:    0x" << buf_size << "\n"
                  << std::dec;
    }

    /* Load input data */
    auto weights = read_file(opts.weights_path);
    auto input   = read_file(opts.input_path);

    /* Verify weight sizes */
    size_t expected_wgate_sz = (size_t)I * H_blk * Q8_BYTES;
    size_t expected_wup_sz   = (size_t)I * H_blk * Q8_BYTES;
    size_t expected_wdown_sz = (size_t)H * I_blk * Q8_BYTES;
    size_t total_expected    = expected_wgate_sz + expected_wup_sz + expected_wdown_sz;

    if (weights.size() < total_expected) {
        std::cerr << "Error: weights file too small. Got " << weights.size()
                  << " bytes, expected at least " << total_expected << "\n";
        return 1;
    }

    if (input.size() != input_sz) {
        std::cerr << "Error: input size mismatch. Got " << input.size()
                  << " bytes, expected " << input_sz << "\n";
        return 1;
    }

    /* Fill parameter block */
    FFNParams params;
    memset(&params, 0, sizeof(params));
    params.input_off    = input_off;
    params.wgate_off    = wgate_off;
    params.wup_off      = wup_off;
    params.wdown_off    = wdown_off;
    params.output_off   = output_off;
    params.scratch_off  = scratch_off;
    params.hidden       = H;
    params.inter        = I;
    params.h_blocks     = H_blk;
    params.i_blocks     = I_blk;

    /* Create device, runtime, stream */
    logging::LoggerDefault logger;

    auto deviceLayer = dev::IDeviceLayer::createPcieDeviceLayer();
    auto runtime = rt::IRuntime::create(std::move(deviceLayer));

    std::atomic<bool> stream_error{false};
    std::atomic<bool> kernel_aborted{false};

    runtime->setOnStreamErrorsCallback([&](rt::EventId, const rt::StreamError& err) {
        stream_error.store(true, std::memory_order_relaxed);
        std::cerr << "Stream error: code " << (int)err.errorCode_ << "\n";
    });
    runtime->setOnKernelAbortedErrorCallback([&](rt::EventId, std::byte*, size_t,
                                                  std::function<void()> freeFn) {
        kernel_aborted.store(true, std::memory_order_relaxed);
        std::cerr << "Kernel aborted\n";
        freeFn();
    });

    auto devices = runtime->getDevices();
    if (devices.empty()) { std::cerr << "No devices found\n"; return 1; }
    auto device = devices[0];
    auto stream = runtime->createStream(device);

    /* Allocate device buffer */
    auto* deviceBuf = runtime->mallocDevice(device, buf_size);

    /* Load kernel ELF */
    auto elf = read_file(opts.elf_path);
    auto loadResult = runtime->loadCode(stream, reinterpret_cast<std::byte*>(elf.data()), elf.size());
    if (!runtime->waitForEvent(loadResult.event_)) {
        std::cerr << "ELF load timed out\n"; return 1;
    }
    std::cout << "Kernel loaded at 0x" << std::hex
              << reinterpret_cast<uintptr_t>(loadResult.loadAddress_) << std::dec << "\n";

    /* Write params to device buffer */
    runtime->memcpyHostToDevice(stream, &params, deviceBuf + param_off, sizeof(params));
    /* Write input */
    runtime->memcpyHostToDevice(stream, input.data(), deviceBuf + input_off, input.size());
    /* Write weights (gate, up, down contiguously) */
    runtime->memcpyHostToDevice(stream, weights.data(), deviceBuf + wgate_off, total_expected);
    runtime->waitForStream(stream, std::chrono::seconds(30));

    if (opts.verbose) {
        std::cout << "Data uploaded to device\n";
    }

    /* Launch kernel */
    rt::KernelLaunchOptions kOpts;
    {
        uint64_t shire_mask = 0;
        for (uint32_t s = 0; s < opts.shires; s++) {
            shire_mask |= (uint64_t{1} << s);
        }
        kOpts.setShireMask(shire_mask);
    }
    kOpts.setBarrier(true);
    kOpts.setFlushL3(true);

    /* arg to kernel main(): pointer to device buffer */
    const uint64_t arg = reinterpret_cast<uintptr_t>(deviceBuf);

    std::cout << "Launching fused FFN kernel (H=" << H << ", I=" << I
              << ", " << opts.shires << " shires)...\n";

    auto launch_start = std::chrono::steady_clock::now();
    runtime->kernelLaunch(stream, loadResult.kernel_,
                          reinterpret_cast<const std::byte*>(&arg), sizeof(arg), kOpts);

    auto timeout = std::chrono::seconds(opts.timeout_secs);
    bool completed = runtime->waitForStream(stream, timeout);
    auto launch_end = std::chrono::steady_clock::now();

    if (!completed) {
        std::cerr << "Kernel timed out\n";
        runtime->abortStream(stream);
        return 1;
    }

    double elapsed = std::chrono::duration<double>(launch_end - launch_start).count();
    std::cout << "Kernel completed in " << elapsed << " seconds\n";

    /* Check for errors */
    auto errors = runtime->retrieveStreamErrors(stream);
    bool has_errors = stream_error.load() || kernel_aborted.load() || !errors.empty();
    if (has_errors) {
        std::cerr << "Kernel reported errors\n";
        for (auto& e : errors) {
            std::cerr << "  error code: " << (int)e.errorCode_ << "\n";
        }
        return 1;
    }

    /* Read back output */
    std::vector<char> output(output_sz);
    auto evt = runtime->memcpyDeviceToHost(stream, deviceBuf + output_off, output.data(), output.size());
    if (!runtime->waitForEvent(evt)) {
        std::cerr << "Output readback timed out\n";
        return 1;
    }

    write_file(opts.output_path, output.data(), output.size());
    std::cout << "Output written to " << opts.output_path << "\n";

    /* Print first few values */
    float* fout = reinterpret_cast<float*>(output.data());
    std::cout << "Output[0..4]:";
    for (int i = 0; i < 5 && i < H; i++) std::cout << " " << fout[i];
    std::cout << "\n";

    /* Cleanup */
    runtime->unloadCode(loadResult.kernel_);
    runtime->freeDevice(device, deviceBuf);
    runtime->destroyStream(stream);

    return 0;
}
