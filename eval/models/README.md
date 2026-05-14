# CNN_V4 Evaluation Step 1: Benchmark Layer Descriptors — v2

This package contains the **main benchmark layer descriptors** for the current CNN_V4 evaluation flow.

## Main descriptors

1. `efficientnet_b0_prefix9_cnn_v4_current.json`
2. `vgg16_conv13.json`
3. `alexnet_conv5.json`
4. `efficientnet_b0_dcp18_blockconv_surrogate.json`
5. `googlenet_v1_inception_conv_ops_expanded.json`

`summary.csv` lists only the active/main descriptors above.

## Excluded descriptor

`MobileNet-v1` is intentionally excluded from the main evaluation plan because canonical MobileNet-v1 uses **depthwise separable convolution**. The current canonical CNN_V4 RTL is treated as a dense Conv/ReLU/Pooling accelerator without native grouped/depthwise convolution support.

The old MobileNet descriptor is kept only for reference at:

```text
excluded/mobilenet_v1_conv28_excluded_depthwise.json
```

Do not use it in the main thesis result table unless native depthwise/grouped-convolution support is added to RTL and the golden/DSE flow.

## Recommended order

1. Start with `efficientnet_b0_prefix9_cnn_v4_current.json` because it mirrors the existing CNN_V4 regression style.
2. Then run `vgg16_conv13.json`.
3. Then run `alexnet_conv5.json`.
4. Use `efficientnet_b0_dcp18_blockconv_surrogate.json` only as a DCP-style surrogate, not as full EfficientNet-B0.
5. Use `googlenet_v1_inception_conv_ops_expanded.json` only after deciding how to handle branch/concat scheduling.

## DSE note

Use the updated `dse_lite_v2.py`. Its default grouped/depthwise policy is now `error`, so accidentally running depthwise MobileNet will fail instead of silently using a dense approximation.
