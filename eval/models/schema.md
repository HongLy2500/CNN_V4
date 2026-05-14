# CNN_V4 Evaluation Layer Descriptor Schema v1

These files are Step 1 inputs for the evaluation/DSE flow.

Important convention:

- `conv_h_out`, `conv_w_out`: convolution output shape before optional pooling.
- `post_h`, `post_w`, `post_c`: logical output shape after optional pooling; this becomes the next layer IFM.
- `ops`: convolution-only operation count, using multiply + add = 2 operations.
- `dse`: intentionally `null` for benchmark descriptors; the DSE step should fill `mode`, `pv_m1`, `pf_m1`, `pc_m2`, and `pf_m2`.
- `ddr`: intentionally `null` in Step 1; DDR layout/addresses are assigned after DSE and packing.
- `cnn_v4_current_rtl_ready`: direct compatibility with the current RTL constraints inferred from source. A warning here does not mean the benchmark is invalid; it means the current RTL may need padding/stride/depthwise/branch handling or a surrogate mapping.

Current CNN_V4 inferred direct constraints:

- Dense standard convolution (`groups=1`).
- ReLU supported.
- 2x2 stride-2 max-pooling supported by current pooling modules.
- Existing regression test `efficientnet_b0_prefix9_cnn_v4_current.json` is the safest descriptor to run first.
