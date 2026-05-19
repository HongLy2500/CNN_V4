#!/usr/bin/env python3
"""
Generate fixed-point golden outputs for CNN_V4 evaluation.

This script runs a software reference model for the Conv/ReLU/Pooling
workloads described by eval/models/*.json, using:
  - ImageNet-derived input tensor saved by prepare_imagenet_inputs.py
  - int8 Conv2d weights exported by export_torchvision_weights.py

It intentionally does NOT compare against TorchVision floating-point outputs.
The generated golden uses the same quantized int8 weights that should be used
by the FPGA run.

Default tensor convention
-------------------------
Input .npy  : uint8 HWC
Weight .npy : int8 [F, C, K, K]
Layer output: uint8 HWC after optional ReLU/store and optional max-pooling
Hex output  : HWC flatten, 4 uint8 values per 32-bit word, little-endian

Important
---------
The produced expected_final_ofm_linear_u32le.hex is a SIMPLE linear HWC pack.
If CNN_V4's OFM DDR layout is tiled/banked/Pv/Pf/Pc-specific, use the final
.npy tensor from this script as the canonical golden tensor and run a separate
RTL-specific packer before comparing with board readback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "ERROR: This script requires torch and numpy.\n"
        "Install them first, for example:\n"
        "  pip install torch numpy\n\n"
        f"Import error: {exc}"
    )


@dataclass
class LayerGoldenRecord:
    layer_id: int
    layer_name: str
    weight_npy: str
    input_shape_hwc: List[int]
    weight_shape_f_c_k_k: List[int]
    conv_shape_hwc: List[int]
    output_shape_hwc: List[int]
    relu_en: bool
    pool_en: bool
    pool_k: int
    pool_stride: int
    conv_stride: int
    padding_top_bottom_left_right: List[int]
    store_policy: str
    output_npy: str
    output_hex_linear_u32le: Optional[str]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    return int(value)


def pack_u8_to_u32_le(values_u8: Iterable[int]) -> List[int]:
    vals = [int(v) & 0xFF for v in values_u8]
    words: List[int] = []

    for i in range(0, len(vals), 4):
        b0 = vals[i + 0] if i + 0 < len(vals) else 0
        b1 = vals[i + 1] if i + 1 < len(vals) else 0
        b2 = vals[i + 2] if i + 2 < len(vals) else 0
        b3 = vals[i + 3] if i + 3 < len(vals) else 0
        words.append(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))

    return words


def write_u32_hex(path: Path, words: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for word in words:
            f.write(f"{int(word) & 0xFFFFFFFF:08X}\n")


def write_hwc_u8_linear_hex(path: Path, tensor_hwc_u8: np.ndarray) -> int:
    if tensor_hwc_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 tensor for hex export, got {tensor_hwc_u8.dtype}")
    words = pack_u8_to_u32_le(tensor_hwc_u8.reshape(-1))
    write_u32_hex(path, words)
    return len(words)


def resolve_path(path_string: str, weights_dir: Path, metadata_path: Optional[Path]) -> Path:
    p = Path(path_string)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(weights_dir / p)
        if metadata_path is not None:
            candidates.append(metadata_path.parent / p)
            candidates.append(metadata_path.parent / p.name)
            candidates.append(metadata_path.parent / "layers" / p.name)

    for cand in candidates:
        if cand.exists():
            return cand

    # Return first candidate for a useful error message.
    return candidates[0] if candidates else p


def discover_weight_files(weights_dir: Path, expected_layers: int) -> Tuple[List[Path], Optional[Path], List[str]]:
    """Return weight_i8 .npy paths in layer order."""
    warnings: List[str] = []
    metadata_path = weights_dir / "metadata.json"

    if metadata_path.exists():
        meta = load_json(metadata_path)
        layers = meta.get("layers", [])
        weight_paths = []

        for layer in layers:
            weight_path_string = layer.get("weight_i8_npy")
            if not weight_path_string:
                continue
            weight_paths.append(resolve_path(str(weight_path_string), weights_dir, metadata_path))

        if len(weight_paths) >= expected_layers:
            return weight_paths[:expected_layers], metadata_path, warnings

        warnings.append(
            f"metadata.json found but provides only {len(weight_paths)} weight_i8 paths; "
            f"expected {expected_layers}. Falling back to glob search."
        )

    candidates = sorted((weights_dir / "layers").glob("*weight_i8.npy"))
    if not candidates:
        candidates = sorted(weights_dir.glob("**/*weight_i8.npy"))

    if len(candidates) < expected_layers:
        raise FileNotFoundError(
            f"Could not find enough int8 weight .npy files in {weights_dir}. "
            f"Found {len(candidates)}, expected {expected_layers}."
        )

    return candidates[:expected_layers], metadata_path if metadata_path.exists() else None, warnings


def validate_descriptor_is_linear(layers: List[Dict[str, Any]], allow_linearized_graph: bool) -> List[str]:
    warnings: List[str] = []
    non_linear = []

    for idx, layer in enumerate(layers):
        branch = layer.get("branch")
        input_from = layer.get("input_from")
        if branch not in (None, "", "null") or input_from not in (None, "", "input", "previous", "prev"):
            # First layer often says input_from=input. Later layer names in graph descriptors are non-linear.
            if idx > 0 or branch not in (None, "", "null"):
                non_linear.append((idx, layer.get("name", f"layer{idx}"), branch, input_from))

    if non_linear and not allow_linearized_graph:
        sample = "\n".join(
            f"  layer {idx} {name}: branch={branch}, input_from={input_from}"
            for idx, name, branch, input_from in non_linear[:10]
        )
        raise ValueError(
            "Descriptor appears to describe a graph/branch workload rather than a simple sequential chain.\n"
            "The current golden script runs layers sequentially: layer i output -> layer i+1 input.\n"
            "Use --allow-linearized-graph only if you intentionally want this approximation.\n"
            f"Examples:\n{sample}"
        )

    if non_linear:
        warnings.append(
            "Descriptor has branch/input_from fields. Running as a linearized sequential workload because "
            "--allow-linearized-graph was set. This is NOT full GoogLeNet graph execution."
        )

    return warnings


def check_groups(layers: List[Dict[str, Any]]) -> None:
    bad = []
    for idx, layer in enumerate(layers):
        groups = as_int(layer.get("groups", 1), 1)
        if groups != 1:
            bad.append((idx, layer.get("name", f"layer{idx}"), groups))
    if bad:
        sample = "\n".join(f"  layer {idx} {name}: groups={groups}" for idx, name, groups in bad[:20])
        raise ValueError(
            "Grouped/depthwise layers are not supported by this dense Conv/ReLU/Pooling golden script.\n"
            f"Examples:\n{sample}"
        )


def store_to_u8(tensor: torch.Tensor, policy: str) -> torch.Tensor:
    """Convert integer-valued tensor to uint8 according to RTL store policy."""
    if policy == "saturate_u8":
        return torch.clamp(tensor, 0, 255).to(torch.uint8)

    if policy == "low8_u8":
        # PyTorch bitwise ops require integer tensors. Round first in case the compute tensor is floating.
        t_i64 = torch.round(tensor).to(torch.int64)
        return torch.bitwise_and(t_i64, 0xFF).to(torch.uint8)

    raise ValueError(f"Unsupported store policy: {policy}")


def expected_conv_hw(h_in: int, w_in: int, k: int, stride: int, pad_t: int, pad_b: int, pad_l: int, pad_r: int) -> Tuple[int, int]:
    h = (h_in + pad_t + pad_b - k) // stride + 1
    w = (w_in + pad_l + pad_r - k) // stride + 1
    return h, w


def run_conv_relu_pool_layer(
    x_hwc_u8: np.ndarray,
    w_fckk_i8: np.ndarray,
    layer: Dict[str, Any],
    store_policy: str,
    compute_dtype: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (conv_or_relu_stored_hwc_u8_before_pool, final_hwc_u8_after_optional_pool)."""
    if x_hwc_u8.dtype != np.uint8:
        raise ValueError(f"Input tensor must be uint8 HWC, got dtype={x_hwc_u8.dtype}")
    if w_fckk_i8.dtype != np.int8:
        raise ValueError(f"Weight tensor must be int8 [F,C,K,K], got dtype={w_fckk_i8.dtype}")

    h_in, w_in, c_in = [int(x) for x in x_hwc_u8.shape]
    f_out, c_w, k_y, k_x = [int(x) for x in w_fckk_i8.shape]
    if c_w != c_in:
        raise ValueError(
            f"Layer {layer.get('name')} channel mismatch: input C={c_in}, weight C={c_w}"
        )
    if k_y != k_x:
        raise ValueError(f"Only square kernels are supported; got {k_y}x{k_x}")

    stride = as_int(layer.get("conv_stride", layer.get("stride", 1)), 1)
    pad_t = as_int(layer.get("pad_top", 0), 0)
    pad_b = as_int(layer.get("pad_bottom", 0), 0)
    pad_l = as_int(layer.get("pad_left", 0), 0)
    pad_r = as_int(layer.get("pad_right", 0), 0)

    dtype = torch.float64 if compute_dtype == "float64" else torch.float32

    # HWC -> NCHW
    x = torch.from_numpy(x_hwc_u8.astype(np.float64 if compute_dtype == "float64" else np.float32))
    x = x.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    w = torch.from_numpy(w_fckk_i8.astype(np.float64 if compute_dtype == "float64" else np.float32))
    w = w.to(device=device, dtype=dtype)

    if pad_t or pad_b or pad_l or pad_r:
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)

    y = F.conv2d(x, w, bias=None, stride=stride, padding=0)

    # Convert to exact integer-valued tensor after float convolution.
    # float64 is exact for the integer ranges used in this evaluation.
    y_int = torch.round(y).to(torch.int64)

    if bool(layer.get("relu_en", True)):
        y_int = torch.clamp(y_int, min=0)

    y_u8 = store_to_u8(y_int, store_policy)
    conv_stored_hwc = y_u8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()

    pool_cfg = layer.get("pool", {}) or {}
    pool_en = bool(pool_cfg.get("enable", False))

    if pool_en:
        pool_k = as_int(pool_cfg.get("k", 2), 2)
        pool_stride = as_int(pool_cfg.get("stride", pool_k), pool_k)
        pool_pad = as_int(pool_cfg.get("pad", 0), 0)
        # Pool operates on the stored 8-bit ReLU/conv output, matching a Conv -> ReLU -> Pooling pipeline.
        y_pool_in = y_u8.to(torch.float64 if compute_dtype == "float64" else torch.float32)
        y_pool = F.max_pool2d(y_pool_in, kernel_size=pool_k, stride=pool_stride, padding=pool_pad)
        y_pool_u8 = store_to_u8(torch.round(y_pool).to(torch.int64), store_policy)
        final_hwc = y_pool_u8.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    else:
        final_hwc = conv_stored_hwc

    return conv_stored_hwc, final_hwc


def validate_layer_shapes(
    layer: Dict[str, Any],
    x_hwc: np.ndarray,
    weight: np.ndarray,
    conv_hwc: np.ndarray,
    final_hwc: np.ndarray,
    strict: bool,
) -> List[str]:
    warnings: List[str] = []
    layer_name = layer.get("name", f"layer{layer.get('id', '?')}")

    exp_h_in = layer.get("h_in")
    exp_w_in = layer.get("w_in")
    exp_c_in = layer.get("c_in")
    if exp_h_in is not None and exp_w_in is not None and exp_c_in is not None:
        expected_input = (int(exp_h_in), int(exp_w_in), int(exp_c_in))
        if tuple(x_hwc.shape) != expected_input:
            msg = f"{layer_name}: input shape {tuple(x_hwc.shape)} != descriptor {expected_input}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

    exp_f = layer.get("f_out")
    exp_c = layer.get("c_in")
    exp_k = layer.get("k")
    if exp_f is not None and exp_c is not None and exp_k is not None:
        expected_w = (int(exp_f), int(exp_c), int(exp_k), int(exp_k))
        if tuple(weight.shape) != expected_w:
            msg = f"{layer_name}: weight shape {tuple(weight.shape)} != descriptor {expected_w}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

    exp_conv_h = layer.get("conv_h_out")
    exp_conv_w = layer.get("conv_w_out")
    if exp_conv_h is not None and exp_conv_w is not None and exp_f is not None:
        expected_conv = (int(exp_conv_h), int(exp_conv_w), int(exp_f))
        if tuple(conv_hwc.shape) != expected_conv:
            msg = f"{layer_name}: conv output shape {tuple(conv_hwc.shape)} != descriptor {expected_conv}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

    exp_post_h = layer.get("post_h", exp_conv_h)
    exp_post_w = layer.get("post_w", exp_conv_w)
    exp_post_c = layer.get("post_c", exp_f)
    if exp_post_h is not None and exp_post_w is not None and exp_post_c is not None:
        expected_final = (int(exp_post_h), int(exp_post_w), int(exp_post_c))
        if tuple(final_hwc.shape) != expected_final:
            msg = f"{layer_name}: final layer output shape {tuple(final_hwc.shape)} != descriptor {expected_final}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate fixed-point golden OFMs for CNN_V4 Conv/ReLU/Pooling benchmarks."
    )
    parser.add_argument("--descriptor", required=True, help="eval/models/*.json benchmark descriptor.")
    parser.add_argument("--input", required=True, help="Input image tensor .npy, expected uint8 HWC.")
    parser.add_argument("--weights-dir", required=True, help="Directory produced by export_torchvision_weights.py.")
    parser.add_argument("--out-dir", required=True, help="Output directory for golden tensors and expected hex.")
    parser.add_argument(
        "--store-policy",
        choices=["saturate_u8", "low8_u8"],
        default="saturate_u8",
        help="How each layer output is converted to uint8 before feeding the next layer.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["float64", "float32"],
        default="float64",
        help="PyTorch convolution dtype. float64 is slower but safer for bit-exact integer accumulation.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device, e.g. cpu or cuda. CPU float64 is recommended for deterministic golden generation.",
    )
    parser.add_argument(
        "--strict-shapes",
        action="store_true",
        help="Fail if descriptor shapes do not match runtime tensor/weight/output shapes.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Run only the first N layers. Useful for debug.",
    )
    parser.add_argument(
        "--save-intermediate",
        choices=["all", "final"],
        default="all",
        help="Save all layer OFMs or only the final OFM.",
    )
    parser.add_argument(
        "--write-layer-hex",
        action="store_true",
        help="Also write a simple linear HWC u32le hex file for each intermediate layer.",
    )
    parser.add_argument(
        "--allow-linearized-graph",
        action="store_true",
        help="Allow descriptors with branch/input_from fields to run as a simple sequential chain. Not full graph execution.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )

    args = parser.parse_args()

    descriptor_path = Path(args.descriptor)
    input_path = Path(args.input)
    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out_dir)

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"ERROR: Output directory already exists and is not empty: {out_dir}\n"
            "Use --overwrite if you want to regenerate outputs."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    desc = load_json(descriptor_path)
    layers: List[Dict[str, Any]] = desc.get("layers", [])
    if not layers:
        raise ValueError(f"Descriptor has no layers: {descriptor_path}")

    if args.max_layers is not None:
        layers = layers[: args.max_layers]

    warnings: List[str] = []
    warnings.extend(validate_descriptor_is_linear(layers, args.allow_linearized_graph))
    check_groups(layers)

    x = np.load(input_path)
    if x.dtype != np.uint8:
        raise ValueError(f"Input .npy must be uint8 HWC. Got dtype={x.dtype}, shape={x.shape}")
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Input .npy must be HWC with 3 channels. Got shape={x.shape}")

    weight_paths, weights_metadata_path, weight_discovery_warnings = discover_weight_files(weights_dir, len(layers))
    warnings.extend(weight_discovery_warnings)

    device = torch.device(args.device)
    records: List[LayerGoldenRecord] = []

    current = x

    print(f"Descriptor : {descriptor_path}")
    print(f"Input      : {input_path}, shape={tuple(current.shape)}, dtype={current.dtype}")
    print(f"Weights dir: {weights_dir}")
    print(f"Output dir : {out_dir}")
    print(f"Layers     : {len(layers)}")
    print(f"Policy     : store={args.store_policy}, compute_dtype={args.compute_dtype}\n")

    for idx, layer in enumerate(layers):
        layer_id = as_int(layer.get("id", idx), idx)
        layer_name = str(layer.get("name", f"layer{idx}"))
        weight_path = weight_paths[idx]
        weight = np.load(weight_path)

        if weight.dtype != np.int8:
            raise ValueError(f"Weight file must contain int8 tensor: {weight_path}, got {weight.dtype}")

        conv_stored, final = run_conv_relu_pool_layer(
            x_hwc_u8=current,
            w_fckk_i8=weight,
            layer=layer,
            store_policy=args.store_policy,
            compute_dtype=args.compute_dtype,
            device=device,
        )

        warnings.extend(
            validate_layer_shapes(
                layer=layer,
                x_hwc=current,
                weight=weight,
                conv_hwc=conv_stored,
                final_hwc=final,
                strict=args.strict_shapes,
            )
        )

        layer_dir = out_dir / f"layer{idx:02d}_{layer_name}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        output_npy_path = layer_dir / "ofm_uint8_hwc.npy"
        layer_hex_path: Optional[Path] = None

        if args.save_intermediate == "all" or idx == len(layers) - 1:
            np.save(output_npy_path, final)
            if args.write_layer_hex:
                layer_hex_path = layer_dir / "ofm_linear_u32le.hex"
                write_hwc_u8_linear_hex(layer_hex_path, final)

        pool_cfg = layer.get("pool", {}) or {}
        record = LayerGoldenRecord(
            layer_id=layer_id,
            layer_name=layer_name,
            weight_npy=str(weight_path),
            input_shape_hwc=[int(v) for v in current.shape],
            weight_shape_f_c_k_k=[int(v) for v in weight.shape],
            conv_shape_hwc=[int(v) for v in conv_stored.shape],
            output_shape_hwc=[int(v) for v in final.shape],
            relu_en=bool(layer.get("relu_en", True)),
            pool_en=bool(pool_cfg.get("enable", False)),
            pool_k=as_int(pool_cfg.get("k", 0), 0),
            pool_stride=as_int(pool_cfg.get("stride", 0), 0),
            conv_stride=as_int(layer.get("conv_stride", 1), 1),
            padding_top_bottom_left_right=[
                as_int(layer.get("pad_top", 0), 0),
                as_int(layer.get("pad_bottom", 0), 0),
                as_int(layer.get("pad_left", 0), 0),
                as_int(layer.get("pad_right", 0), 0),
            ],
            store_policy=args.store_policy,
            output_npy=str(output_npy_path) if (args.save_intermediate == "all" or idx == len(layers) - 1) else "",
            output_hex_linear_u32le=str(layer_hex_path) if layer_hex_path is not None else None,
        )
        records.append(record)

        print(
            f"[OK] layer {idx:02d} {layer_name}: "
            f"input={tuple(record.input_shape_hwc)} weight={tuple(record.weight_shape_f_c_k_k)} "
            f"conv={tuple(record.conv_shape_hwc)} final={tuple(record.output_shape_hwc)}"
        )

        current = final

    final_npy = out_dir / "golden_final_ofm_uint8_hwc.npy"
    final_hex = out_dir / "expected_final_ofm_linear_u32le.hex"
    np.save(final_npy, current)
    final_words = write_hwc_u8_linear_hex(final_hex, current)

    metadata = {
        "script": Path(__file__).name,
        "descriptor": str(descriptor_path),
        "model_name": desc.get("model_name", desc.get("name")),
        "input_npy": str(input_path),
        "input_shape_hwc": [int(v) for v in x.shape],
        "weights_dir": str(weights_dir),
        "weights_metadata": str(weights_metadata_path) if weights_metadata_path is not None else None,
        "num_layers_run": len(layers),
        "store_policy": args.store_policy,
        "compute_dtype": args.compute_dtype,
        "device": args.device,
        "output_tensor_layout": "HWC",
        "output_dtype": "uint8",
        "linear_hex_pack": {
            "file": str(final_hex),
            "flatten_order": "HWC NumPy row-major",
            "word_format": "4 uint8 bytes per 32-bit word, little-endian",
            "num_u32_words": int(final_words),
            "warning": "This is a simple linear pack. Use an RTL-specific OFM packer if CNN_V4 writes DDR in a tiled/banked layout.",
        },
        "files": {
            "golden_final_ofm_uint8_hwc_npy": str(final_npy),
            "expected_final_ofm_linear_u32le_hex": str(final_hex),
        },
        "warnings": warnings,
        "layers": [asdict(r) for r in records],
    }

    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    print("\nDone.")
    print(f"  Final OFM shape : {tuple(current.shape)}")
    print(f"  Final OFM npy   : {final_npy}")
    print(f"  Final OFM hex   : {final_hex}")
    print(f"  Metadata        : {metadata_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
