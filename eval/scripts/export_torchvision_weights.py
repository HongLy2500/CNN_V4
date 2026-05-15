#!/usr/bin/env python3
"""
Export convolution weights from TorchVision models for CNN_V4 evaluation.

Purpose
-------
This script extracts Conv2d weights from a TorchVision pretrained model,
quantizes them to int8, and writes both tensor files (.npy) and a simple
linear DDR hex file. The generated int8 weights must be used by BOTH:
  1. the fixed-point golden model, and
  2. the FPGA/RTL input path.

Important
---------
The default DDR packing in this script is a simple linear pack:
  tensor layout: [F, C, K, K]
  flatten order : NumPy row-major over [F, C, KY, KX]
  word format   : 4 int8 bytes per 32-bit word, little-endian

If CNN_V4's weight DDR layout expects a Ptotal/block/tile-specific order,
use the .npy files produced by this script as the canonical tensor source,
then run a separate RTL-specific packer to generate the final DDR hex.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
except Exception as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "ERROR: This script requires torch and torchvision.\n"
        "Install them first, for example:\n"
        "  pip install torch torchvision numpy\n\n"
        f"Import error: {exc}"
    )


SUPPORTED_MODELS = {
    "vgg16": {
        "ctor": "vgg16",
        "weights_enum": "VGG16_Weights",
        "default_out_dir": "eval/assets/vgg16_conv13/weights",
        "note": "Dense Conv2d feature extractor; recommended first benchmark.",
    },
    "alexnet": {
        "ctor": "alexnet",
        "weights_enum": "AlexNet_Weights",
        "default_out_dir": "eval/assets/alexnet_conv5/weights",
        "note": "Dense Conv2d feature extractor; useful for K=11/K=5/K=3.",
    },
    "googlenet": {
        "ctor": "googlenet",
        "weights_enum": "GoogLeNet_Weights",
        "default_out_dir": "eval/assets/googlenet_v1_inception_conv_ops_expanded/weights",
        "note": "Contains Inception branches. Exported Conv2d order follows PyTorch module traversal; use carefully.",
    },
    "efficientnet_b0": {
        "ctor": "efficientnet_b0",
        "weights_enum": "EfficientNet_B0_Weights",
        "default_out_dir": "eval/assets/efficientnet_b0_torchvision/weights",
        "note": "Contains depthwise/grouped layers, SE, residuals, SiLU/BN. Not directly compatible with dense-only CNN_V4 RTL unless you have a compatible mapping.",
    },
}


@dataclass
class ExportedLayer:
    conv_id: int
    module_name: str
    module_type: str
    shape_f_c_k_k: List[int]
    groups: int
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    bias_present: bool
    quantization: str
    scale: Optional[float]
    zero_point: int
    offset_words: int
    offset_bytes: int
    num_values: int
    num_bytes_unpadded: int
    num_words_packed: int
    num_padding_bytes: int
    weight_float_npy: str
    weight_i8_npy: str
    weight_i8_hex: str
    bias_float_npy: Optional[str]


def sanitize_name(name: str) -> str:
    name = name.replace(".", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name.strip("_") or "module"


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


def write_i8_byte_hex(path: Path, values_i8: np.ndarray) -> None:
    """Write one signed int8 value per line as two hex digits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = values_i8.reshape(-1)
    with path.open("w", encoding="utf-8") as f:
        for value in flat:
            f.write(f"{int(value) & 0xFF:02X}\n")


def quantize_symmetric_i8_per_tensor(w_float: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Symmetric per-tensor int8 quantization.

    q = round(w / scale), scale = max(abs(w)) / 127.
    The result is clipped to signed int8 range [-128, 127].
    """
    max_abs = float(np.max(np.abs(w_float))) if w_float.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
    q = np.round(w_float / scale)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scale


def quantize_none_cast_i8(w_float: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    """
    Directly round and clip float weights to int8 without scale metadata.

    This is mainly for debugging; pretrained float weights are usually small,
    so this option may collapse many values to zero.
    """
    q = np.round(w_float)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, None


def load_descriptor_layers(descriptor_path: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    if descriptor_path is None:
        return None

    with descriptor_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "layers" in data and isinstance(data["layers"], list):
        return data["layers"]

    if isinstance(data, list):
        return data

    raise ValueError(f"Cannot find layers list in descriptor: {descriptor_path}")


def descriptor_layer_shape(layer: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (F, C, K) from a flexible benchmark descriptor layer."""
    f = layer.get("f_out", layer.get("f", layer.get("F")))
    c = layer.get("c_in", layer.get("c", layer.get("C")))
    k = layer.get("k", layer.get("K", layer.get("kernel", layer.get("kernel_size"))))

    if isinstance(k, list):
        if len(k) == 2 and k[0] == k[1]:
            k = k[0]
        else:
            k = None

    def to_int_or_none(x: Any) -> Optional[int]:
        if x is None:
            return None
        try:
            return int(x)
        except Exception:
            return None

    return to_int_or_none(f), to_int_or_none(c), to_int_or_none(k)


def validate_against_descriptor(
    convs: List[Tuple[str, nn.Conv2d]],
    descriptor_layers: Optional[List[Dict[str, Any]]],
    strict: bool,
) -> List[str]:
    warnings: List[str] = []
    if descriptor_layers is None:
        return warnings

    if len(convs) != len(descriptor_layers):
        msg = (
            f"Descriptor has {len(descriptor_layers)} layers, but TorchVision model traversal found "
            f"{len(convs)} Conv2d modules."
        )
        if strict:
            raise ValueError(msg)
        warnings.append(msg)

    n = min(len(convs), len(descriptor_layers))
    for idx in range(n):
        module_name, conv = convs[idx]
        desc = descriptor_layers[idx]
        exp_f, exp_c, exp_k = descriptor_layer_shape(desc)
        got_f, got_c, got_ky, got_kx = tuple(int(x) for x in conv.weight.shape)

        mismatches = []
        if exp_f is not None and exp_f != got_f:
            mismatches.append(f"F descriptor={exp_f}, model={got_f}")
        if exp_c is not None and exp_c != got_c:
            mismatches.append(f"C descriptor={exp_c}, model={got_c}")
        if exp_k is not None and (exp_k != got_ky or exp_k != got_kx):
            mismatches.append(f"K descriptor={exp_k}, model={got_ky}x{got_kx}")

        if mismatches:
            layer_name = desc.get("name", desc.get("layer_name", f"layer{idx}"))
            msg = f"Layer {idx} descriptor '{layer_name}' vs module '{module_name}': " + "; ".join(mismatches)
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

    return warnings


def get_weight_enum_value(model_name: str, weights_name: str) -> Any:
    cfg = SUPPORTED_MODELS[model_name]
    enum_name = cfg["weights_enum"]
    enum_cls = getattr(tv_models, enum_name)

    if weights_name.upper() == "DEFAULT":
        return enum_cls.DEFAULT

    try:
        return getattr(enum_cls, weights_name)
    except AttributeError as exc:
        valid = [x for x in dir(enum_cls) if x.isupper()]
        raise ValueError(
            f"Unknown weights '{weights_name}' for {model_name}. "
            f"Valid examples: DEFAULT, {', '.join(valid)}"
        ) from exc


def load_torchvision_model(model_name: str, weights_name: str, no_pretrained: bool) -> nn.Module:
    cfg = SUPPORTED_MODELS[model_name]
    ctor = getattr(tv_models, cfg["ctor"])

    if no_pretrained:
        kwargs: Dict[str, Any] = {"weights": None}
    else:
        weights = get_weight_enum_value(model_name, weights_name)
        kwargs = {"weights": weights}

    # GoogLeNet in TorchVision can expose aux classifiers. Disable output aux heads
    # for inference, but weights can still be loaded from the official checkpoint.
    if model_name == "googlenet":
        kwargs.setdefault("aux_logits", True)

    model = ctor(**kwargs)
    model.eval()
    return model


def collect_conv_modules(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    convs: List[Tuple[str, nn.Conv2d]] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            convs.append((module_name, module))
    return convs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export TorchVision Conv2d weights to int8 .npy files and simple packed .hex for CNN_V4 evaluation."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(SUPPORTED_MODELS.keys()),
        help="TorchVision model to export.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. If omitted, a model-specific eval/assets path is used.",
    )
    parser.add_argument(
        "--descriptor",
        default=None,
        help="Optional eval/models/*.json descriptor to validate Conv2d shapes/order.",
    )
    parser.add_argument(
        "--strict-descriptor",
        action="store_true",
        help="Fail if descriptor layer count or shapes do not match exported Conv2d layers.",
    )
    parser.add_argument(
        "--weights",
        default="DEFAULT",
        help="TorchVision weights enum value, usually DEFAULT or IMAGENET1K_V1.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load pretrained weights. Mostly useful for debugging the export flow.",
    )
    parser.add_argument(
        "--quant",
        choices=["symmetric_i8_per_tensor", "none_cast_i8"],
        default="symmetric_i8_per_tensor",
        help="Quantization method for float weights.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Export only the first N Conv2d layers.",
    )
    parser.add_argument(
        "--allow-grouped",
        action="store_true",
        help="Allow grouped/depthwise Conv2d export. By default groups != 1 is an error.",
    )
    parser.add_argument(
        "--export-bias",
        action="store_true",
        help="Export Conv2d bias .npy files when present. Bias is not packed into all_weights hex.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir or SUPPORTED_MODELS[args.model]["default_out_dir"])
    layers_dir = out_dir / "layers"

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"ERROR: Output directory already exists and is not empty: {out_dir}\n"
            "Use --overwrite if you want to replace/update generated files."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    layers_dir.mkdir(parents=True, exist_ok=True)

    model = load_torchvision_model(
        model_name=args.model,
        weights_name=args.weights,
        no_pretrained=args.no_pretrained,
    )
    convs = collect_conv_modules(model)

    if args.max_layers is not None:
        convs = convs[: args.max_layers]

    grouped = [(name, conv.groups, tuple(int(x) for x in conv.weight.shape)) for name, conv in convs if conv.groups != 1]
    if grouped and not args.allow_grouped:
        msg_lines = [
            "ERROR: Found grouped/depthwise Conv2d layers, but --allow-grouped was not set.",
            "CNN_V4 dense-conv evaluation should not silently export grouped/depthwise layers.",
            "Grouped layers found:",
        ]
        for name, groups, shape in grouped[:20]:
            msg_lines.append(f"  {name}: groups={groups}, weight_shape={shape}")
        if len(grouped) > 20:
            msg_lines.append(f"  ... and {len(grouped) - 20} more")
        msg_lines.append("Use --allow-grouped only if your RTL/golden flow supports this mapping.")
        raise SystemExit("\n".join(msg_lines))

    descriptor_layers = load_descriptor_layers(Path(args.descriptor)) if args.descriptor else None
    descriptor_warnings = validate_against_descriptor(convs, descriptor_layers, args.strict_descriptor)

    all_words: List[int] = []
    exported_layers: List[ExportedLayer] = []
    offsets_rows: List[Dict[str, Any]] = []

    for conv_id, (module_name, conv) in enumerate(convs):
        safe_name = sanitize_name(module_name)
        prefix = f"conv{conv_id:02d}_{safe_name}"

        w_float = conv.weight.detach().cpu().numpy().astype(np.float32)

        if args.quant == "symmetric_i8_per_tensor":
            w_i8, scale = quantize_symmetric_i8_per_tensor(w_float)
        elif args.quant == "none_cast_i8":
            w_i8, scale = quantize_none_cast_i8(w_float)
        else:  # defensive guard
            raise ValueError(f"Unsupported quantization method: {args.quant}")

        weight_float_npy = layers_dir / f"{prefix}_weight_float.npy"
        weight_i8_npy = layers_dir / f"{prefix}_weight_i8.npy"
        weight_i8_hex = layers_dir / f"{prefix}_weight_i8_bytes.hex"

        np.save(weight_float_npy, w_float)
        np.save(weight_i8_npy, w_i8)
        write_i8_byte_hex(weight_i8_hex, w_i8)

        bias_float_npy: Optional[Path] = None
        if args.export_bias and conv.bias is not None:
            bias_float_npy = layers_dir / f"{prefix}_bias_float.npy"
            np.save(bias_float_npy, conv.bias.detach().cpu().numpy().astype(np.float32))

        offset_words = len(all_words)
        offset_bytes = offset_words * 4

        byte_values = [int(v) & 0xFF for v in w_i8.reshape(-1)]
        words = pack_u8_to_u32_le(byte_values)
        all_words.extend(words)

        num_values = int(w_i8.size)
        num_bytes_unpadded = num_values
        num_words_packed = len(words)
        num_padding_bytes = (num_words_packed * 4) - num_bytes_unpadded

        layer_record = ExportedLayer(
            conv_id=conv_id,
            module_name=module_name,
            module_type="torch.nn.Conv2d",
            shape_f_c_k_k=[int(x) for x in w_i8.shape],
            groups=int(conv.groups),
            stride=[int(x) for x in conv.stride],
            padding=[int(x) for x in conv.padding],
            dilation=[int(x) for x in conv.dilation],
            bias_present=conv.bias is not None,
            quantization=args.quant,
            scale=float(scale) if scale is not None else None,
            zero_point=0,
            offset_words=offset_words,
            offset_bytes=offset_bytes,
            num_values=num_values,
            num_bytes_unpadded=num_bytes_unpadded,
            num_words_packed=num_words_packed,
            num_padding_bytes=num_padding_bytes,
            weight_float_npy=str(weight_float_npy),
            weight_i8_npy=str(weight_i8_npy),
            weight_i8_hex=str(weight_i8_hex),
            bias_float_npy=str(bias_float_npy) if bias_float_npy is not None else None,
        )
        exported_layers.append(layer_record)

        offsets_rows.append(
            {
                "conv_id": conv_id,
                "module_name": module_name,
                "shape_f_c_k_k": "x".join(str(x) for x in w_i8.shape),
                "groups": int(conv.groups),
                "offset_words": offset_words,
                "offset_bytes": offset_bytes,
                "num_words_packed": num_words_packed,
                "num_bytes_unpadded": num_bytes_unpadded,
                "num_padding_bytes": num_padding_bytes,
                "scale": float(scale) if scale is not None else "",
            }
        )

        print(
            f"[OK] conv{conv_id:02d} {module_name}: shape={tuple(w_i8.shape)}, "
            f"groups={conv.groups}, offset_words={offset_words}, words={num_words_packed}"
        )

    all_weights_hex = out_dir / "all_weights_linear_u32le.hex"
    write_u32_hex(all_weights_hex, all_words)

    offsets_csv = out_dir / "weight_offsets.csv"
    with offsets_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "conv_id",
            "module_name",
            "shape_f_c_k_k",
            "groups",
            "offset_words",
            "offset_bytes",
            "num_words_packed",
            "num_bytes_unpadded",
            "num_padding_bytes",
            "scale",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(offsets_rows)

    metadata = {
        "script": Path(__file__).name,
        "model": args.model,
        "model_note": SUPPORTED_MODELS[args.model]["note"],
        "weights": None if args.no_pretrained else args.weights,
        "no_pretrained": bool(args.no_pretrained),
        "quantization": {
            "method": args.quant,
            "scope": "per Conv2d tensor" if args.quant == "symmetric_i8_per_tensor" else "direct cast",
            "dtype": "int8",
            "zero_point": 0,
            "note": "Use these quantized weights for both golden model and FPGA run.",
        },
        "tensor_layout": "F,C,K,K",
        "linear_hex_pack": {
            "file": str(all_weights_hex),
            "flatten_order": "NumPy row-major over F,C,KY,KX per layer, layers concatenated in PyTorch named_modules Conv2d traversal order",
            "word_format": "4 int8 bytes per 32-bit word, little-endian",
            "warning": "This is a simple linear pack. If CNN_V4 weight DDR layout is Ptotal/block-specific, run a separate RTL-specific packer using the .npy files.",
        },
        "descriptor": str(args.descriptor) if args.descriptor else None,
        "descriptor_warnings": descriptor_warnings,
        "allow_grouped": bool(args.allow_grouped),
        "num_conv_layers_exported": len(exported_layers),
        "total_packed_words": len(all_words),
        "total_packed_bytes": len(all_words) * 4,
        "files": {
            "all_weights_linear_u32le_hex": str(all_weights_hex),
            "weight_offsets_csv": str(offsets_csv),
            "layers_dir": str(layers_dir),
        },
        "layers": [asdict(x) for x in exported_layers],
    }

    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if descriptor_warnings:
        print("\nDescriptor warnings:")
        for warning in descriptor_warnings:
            print(f"  WARNING: {warning}")

    print("\nDone.")
    print(f"  Conv layers exported : {len(exported_layers)}")
    print(f"  Packed weight hex    : {all_weights_hex}")
    print(f"  Offsets CSV          : {offsets_csv}")
    print(f"  Metadata             : {metadata_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
