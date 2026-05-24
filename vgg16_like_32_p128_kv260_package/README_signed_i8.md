# SIGNED INT8 CPU baseline helper — old *_uint8_hwc.npy filenames supported as primary

This revision matches the current local package convention: the input/golden files may still be named:

```text
input_uint8_hwc.npy
golden_final_ofm_uint8_hwc.npy
```

but their contents are treated as **signed INT8 / two's-complement bytes**. The exporter now searches these old names first, then falls back to newer `*_i8_hwc.npy` names. The C++ baseline always interprets input, output, and weights as signed `int8_t` values.

## Files

```text
export_cpu_baseline_signed_from_pkg.py
cpu_baseline_signed_i8.cpp
```

## Usage

Copy both files into:

```text
D:\sdh\CNN_V4\vgg16_like_32_p128_kv260_package
```

Then run:

```powershell
cd D:\sdh\CNN_V4\vgg16_like_32_p128_kv260_package

python .\export_cpu_baseline_signed_from_pkg.py `
  --pkg-root . `
  --bench vgg16_like_32_p128 `
  --image img0000 `
  --out cpu_baseline_signed_run
```

The script will explicitly prefer:

```text
eval\assets\vgg16_like_32_p128\inputs\img0000\input_uint8_hwc.npy
eval\golden\vgg16_like_32_p128\img0000\golden_final_ofm_uint8_hwc.npy
```

and will export raw signed-byte files:

```text
cpu_baseline_signed_run\ifm_l0_i8_hwc.bin
cpu_baseline_signed_run\expected_final_ofm_i8_hwc.bin
cpu_baseline_signed_run\w_l0_i8.bin ... w_l12_i8.bin
cpu_baseline_signed_run\b_l0_i32.bin ... b_l12_i32.bin
cpu_baseline_signed_run\layers_cpu.csv
```

Compile and run:

```powershell
cd .\cpu_baseline_signed_run
copy ..\cpu_baseline_signed_i8.cpp .
g++ -O3 -std=c++17 .\cpu_baseline_signed_i8.cpp -o cpu_baseline_signed_i8.exe
.\cpu_baseline_signed_i8.exe 1000
```

Expected output includes:

```text
CPU_PASS
CPU_AVG_LATENCY_MS=...
OPS_CONV_ONLY=28901376
GOPS_CONV_ONLY=...
SIGNED_BASELINE=1
```

If your fixed-point golden uses an output rescale/shift, compile with:

```powershell
g++ -O3 -std=c++17 -DOUT_SHIFT=<N> .\cpu_baseline_signed_i8.cpp -o cpu_baseline_signed_i8.exe
```

If it is only signed saturate with no rescale, keep default `OUT_SHIFT=0`.
