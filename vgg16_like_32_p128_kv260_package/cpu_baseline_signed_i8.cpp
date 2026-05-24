// cpu_baseline_signed_i8.cpp
// CPU baseline for CNN_V4 generated package using SIGNED 8-bit fixed-point.
//
// Data convention:
//   input  : int8 HWC bytes, file ifm_l0_i8_hwc.bin
//   weight : int8 F,C,K,K bytes, files w_lN_i8.bin
//   bias   : int32 per output channel, files b_lN_i32.bin (zeros if absent)
//   compute: int32 accumulation
//   store  : signed saturation to int8 [-128,127]
//   ReLU   : if enabled, clamp accumulator to >=0 before signed saturation
//   pool   : maxpool 2x2 stride 2, signed int8 comparison
//
// Compile:
//   g++ -O3 -std=c++17 cpu_baseline_signed_i8.cpp -o cpu_baseline_signed_i8
// Optional output shift if your RTL/golden uses fixed-point rescale:
//   g++ -O3 -std=c++17 -DOUT_SHIFT=8 cpu_baseline_signed_i8.cpp -o cpu_baseline_signed_i8

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef OUT_SHIFT
#define OUT_SHIFT 0
#endif

struct Layer {
    int layer_id = 0;
    std::string name;
    int h = 0, w = 0, c = 0, f = 0, k = 0;
    int pad_t = 0, pad_b = 0, pad_l = 0, pad_r = 0;
    int relu = 1;
    int pool = 0;
    int pool_k = 2;
    int pool_stride = 2;
    int h_out = 0, w_out = 0;
};

struct Tensor {
    int h = 0, w = 0, c = 0;
    std::vector<uint8_t> data;  // raw bytes; interpret through s8()

    uint8_t& at_raw(int y, int x, int ch) {
        return data[(static_cast<size_t>(y) * w + x) * c + ch];
    }
    uint8_t at_raw(int y, int x, int ch) const {
        return data[(static_cast<size_t>(y) * w + x) * c + ch];
    }
};

static inline int s8(uint8_t b) {
    return static_cast<int>(static_cast<int8_t>(b));
}

static inline uint8_t s8_to_raw(int32_t x) {
    if (x < -128) x = -128;
    if (x > 127) x = 127;
    return static_cast<uint8_t>(static_cast<int8_t>(x));
}

static inline int32_t maybe_shift(int32_t x) {
#if OUT_SHIFT > 0
    // C++ right shift of negative signed integers is implementation-defined before C++20,
    // but all usual compilers for x86/AArch64 implement arithmetic shift. This matches RTL signed >>>.
    return x >> OUT_SHIFT;
#elif OUT_SHIFT < 0
    return x << (-OUT_SHIFT);
#else
    return x;
#endif
}

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(tok);
    return out;
}

static int parse_int(const std::string& s, int def = 0) {
    if (s.empty()) return def;
    return std::stoi(s, nullptr, 0);
}

static std::vector<Layer> read_layers(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open " + path);

    std::string header, line;
    std::getline(f, header);
    auto keys = split_csv_line(header);

    std::vector<Layer> layers;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto vals = split_csv_line(line);
        auto get = [&](const std::string& key) -> std::string {
            for (size_t i = 0; i < keys.size(); ++i) {
                if (keys[i] == key) return i < vals.size() ? vals[i] : "";
            }
            return "";
        };

        Layer L;
        L.layer_id = parse_int(get("layer_id"), static_cast<int>(layers.size()));
        L.name = get("layer_name");
        L.h = parse_int(get("h_in"));
        L.w = parse_int(get("w_in"));
        L.c = parse_int(get("c_in"));
        L.f = parse_int(get("f_out"));
        L.k = parse_int(get("k"), 3);
        L.pad_t = parse_int(get("pad_top"));
        L.pad_b = parse_int(get("pad_bottom"));
        L.pad_l = parse_int(get("pad_left"));
        L.pad_r = parse_int(get("pad_right"));
        L.relu = parse_int(get("relu_en"), 1);
        L.pool = parse_int(get("pool_en"), 0);
        L.pool_k = parse_int(get("pool_k"), 2);
        L.pool_stride = parse_int(get("pool_stride"), 2);
        L.h_out = parse_int(get("h_out"));
        L.w_out = parse_int(get("w_out"));
        layers.push_back(L);
    }
    return layers;
}

static std::vector<uint8_t> read_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::vector<int32_t> read_i32_raw(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    std::vector<int32_t> data(count, 0);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(int32_t)));
    if (static_cast<size_t>(f.gcount()) != count * sizeof(int32_t)) {
        throw std::runtime_error("Short read for " + path);
    }
    return data;
}

static Tensor conv_relu_store_signed(
    const Tensor& in,
    const std::vector<uint8_t>& wraw,
    const std::vector<int32_t>& bias,
    const Layer& L
) {
    const int ho = (L.h + L.pad_t + L.pad_b - L.k) + 1;
    const int wo = (L.w + L.pad_l + L.pad_r - L.k) + 1;

    if (in.h != L.h || in.w != L.w || in.c != L.c) {
        std::ostringstream oss;
        oss << "Layer input shape mismatch at L" << L.layer_id
            << ": got HWC=" << in.h << "x" << in.w << "x" << in.c
            << ", expected " << L.h << "x" << L.w << "x" << L.c;
        throw std::runtime_error(oss.str());
    }

    const size_t expected_w = static_cast<size_t>(L.f) * L.c * L.k * L.k;
    if (wraw.size() != expected_w) {
        throw std::runtime_error("Weight size mismatch at L" + std::to_string(L.layer_id));
    }
    if (bias.size() != static_cast<size_t>(L.f)) {
        throw std::runtime_error("Bias size mismatch at L" + std::to_string(L.layer_id));
    }

    Tensor out;
    out.h = ho;
    out.w = wo;
    out.c = L.f;
    out.data.assign(static_cast<size_t>(ho) * wo * L.f, 0);

    auto w_at = [&](int f, int c, int ky, int kx) -> int {
        size_t idx = (((static_cast<size_t>(f) * L.c + c) * L.k + ky) * L.k + kx);
        return s8(wraw[idx]);
    };

    for (int y = 0; y < ho; ++y) {
        for (int x = 0; x < wo; ++x) {
            for (int f = 0; f < L.f; ++f) {
                int32_t acc = bias[f];

                for (int c = 0; c < L.c; ++c) {
                    for (int ky = 0; ky < L.k; ++ky) {
                        const int iy = y + ky - L.pad_t;
                        if (iy < 0 || iy >= L.h) continue;
                        for (int kx = 0; kx < L.k; ++kx) {
                            const int ix = x + kx - L.pad_l;
                            if (ix < 0 || ix >= L.w) continue;
                            acc += static_cast<int32_t>(s8(in.at_raw(iy, ix, c))) * w_at(f, c, ky, kx);
                        }
                    }
                }

                acc = maybe_shift(acc);
                if (L.relu && acc < 0) acc = 0;
                out.at_raw(y, x, f) = s8_to_raw(acc);
            }
        }
    }
    return out;
}

static Tensor maxpool_signed(const Tensor& in, const Layer& L) {
    if (!L.pool) return in;
    if (L.pool_k != 2 || L.pool_stride != 2) {
        throw std::runtime_error("Only maxpool 2x2 stride 2 is implemented");
    }
    Tensor out;
    out.h = in.h / 2;
    out.w = in.w / 2;
    out.c = in.c;
    out.data.assign(static_cast<size_t>(out.h) * out.w * out.c, 0);

    for (int y = 0; y < out.h; ++y) {
        for (int x = 0; x < out.w; ++x) {
            for (int c = 0; c < out.c; ++c) {
                uint8_t best_raw = in.at_raw(2*y, 2*x, c);
                int best = s8(best_raw);
                const uint8_t candidates[3] = {
                    in.at_raw(2*y, 2*x + 1, c),
                    in.at_raw(2*y + 1, 2*x, c),
                    in.at_raw(2*y + 1, 2*x + 1, c),
                };
                for (uint8_t raw : candidates) {
                    int v = s8(raw);
                    if (v > best) {
                        best = v;
                        best_raw = raw;
                    }
                }
                out.at_raw(y, x, c) = best_raw;
            }
        }
    }
    return out;
}

static Tensor run_once(
    const std::vector<Layer>& layers,
    const Tensor& input,
    const std::vector<std::vector<uint8_t>>& weights,
    const std::vector<std::vector<int32_t>>& biases
) {
    Tensor cur = input;
    for (const Layer& L : layers) {
        Tensor conv = conv_relu_store_signed(cur, weights.at(L.layer_id), biases.at(L.layer_id), L);
        cur = maxpool_signed(conv, L);
    }
    return cur;
}

static uint64_t conv_ops(const std::vector<Layer>& layers) {
    uint64_t ops = 0;
    for (const Layer& L : layers) {
        int ho = (L.h + L.pad_t + L.pad_b - L.k) + 1;
        int wo = (L.w + L.pad_l + L.pad_r - L.k) + 1;
        ops += 2ull * ho * wo * L.f * L.c * L.k * L.k;
    }
    return ops;
}

int main(int argc, char** argv) {
    try {
        int repeat = 1000;
        if (argc >= 2) repeat = std::max(1, std::atoi(argv[1]));

        const auto layers = read_layers("layers_cpu.csv");
        if (layers.empty()) throw std::runtime_error("No layers in layers_cpu.csv");

        Tensor input;
        input.h = layers[0].h;
        input.w = layers[0].w;
        input.c = layers[0].c;
        input.data = read_raw("ifm_l0_i8_hwc.bin");
        const size_t input_exp = static_cast<size_t>(input.h) * input.w * input.c;
        if (input.data.size() != input_exp) {
            throw std::runtime_error("Input size mismatch: got " + std::to_string(input.data.size()) +
                                     ", expected " + std::to_string(input_exp));
        }

        std::vector<std::vector<uint8_t>> weights(layers.size());
        std::vector<std::vector<int32_t>> biases(layers.size());
        for (const auto& L : layers) {
            weights.at(L.layer_id) = read_raw("w_l" + std::to_string(L.layer_id) + "_i8.bin");
            biases.at(L.layer_id) = read_i32_raw("b_l" + std::to_string(L.layer_id) + "_i32.bin", L.f);
        }

        const auto expected = read_raw("expected_final_ofm_i8_hwc.bin");
        const Tensor out = run_once(layers, input, weights, biases);

        if (out.data.size() != expected.size()) {
            std::cout << "CPU_FAIL length mismatch got=" << out.data.size()
                      << " expected=" << expected.size() << "\n";
            return 1;
        }

        int mism = 0;
        for (size_t i = 0; i < out.data.size(); ++i) {
            if (out.data[i] != expected[i]) {
                if (mism < 20) {
                    std::cout << "MISMATCH i=" << i
                              << " got_raw=0x" << std::hex << (static_cast<int>(out.data[i]) & 0xFF)
                              << " exp_raw=0x" << (static_cast<int>(expected[i]) & 0xFF) << std::dec
                              << " got_s8=" << s8(out.data[i])
                              << " exp_s8=" << s8(expected[i]) << "\n";
                }
                ++mism;
            }
        }
        if (mism != 0) {
            std::cout << "CPU_FAIL mismatch=" << mism << "\n";
            return 1;
        }
        std::cout << "CPU_PASS\n";

        volatile uint64_t checksum = 0;
        {
            Tensor tmp = run_once(layers, input, weights, biases);
            for (uint8_t b : tmp.data) checksum += static_cast<uint64_t>(b);
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; ++i) {
            Tensor tmp = run_once(layers, input, weights, biases);
            checksum += tmp.data.empty() ? 0 : tmp.data[0];
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double avg_ms = total_ms / static_cast<double>(repeat);
        const uint64_t ops = conv_ops(layers);
        const double gops = (static_cast<double>(ops) / (avg_ms / 1000.0)) / 1e9;

        std::cout << "CPU_REPEAT=" << repeat << "\n";
        std::cout << "CPU_TOTAL_LATENCY_MS=" << total_ms << "\n";
        std::cout << "CPU_AVG_LATENCY_MS=" << avg_ms << "\n";
        std::cout << "OPS_CONV_ONLY=" << ops << "\n";
        std::cout << "GOPS_CONV_ONLY=" << gops << "\n";
        std::cout << "OUT_SHIFT=" << OUT_SHIFT << "\n";
        std::cout << "SIGNED_BASELINE=1\n";
        std::cout << "CHECKSUM_IGNORE=" << checksum << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }
}
