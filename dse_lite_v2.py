#!/usr/bin/env python3
"""
dse_lite_v2.py - Offline DSE-lite for CNN_V4 benchmark layer descriptors.

Purpose
-------
Read a Step-1 benchmark layer descriptor JSON and choose a DCP-CNN-style
parallelism schedule:

  Mode 1: dynamic pixel/filter parallelism per layer
          Pv_i * Pf_i = Ptotal

  Mode 2: fixed channel/filter parallelism for the suffix
          Pc * Pf = Ptotal

The script searches all possible transition layers and factor pairs, then
exports a CSV schedule and, optionally, an updated JSON descriptor with the
`dse` fields filled.

This is intentionally an offline estimation tool. It does not program the FPGA
and does not replace cycle counters measured from the real RTL.

Descriptor schema expected
--------------------------
Each layer should have at least:
  id, name, h_in, w_in, c_in, f_out, k, conv_h_out, conv_w_out
Optional:
  op_type, groups, conv_stride, pool, ops, cnn_v4_current_rtl_ready

Example
-------
python eval/dse/dse_lite.py \
  --model eval/models/efficientnet_b0_prefix9_cnn_v4_current.json \
  --ptotal 2048 \
  --pv-max 128 \
  --pf-max 128 \
  --pc-max 32 \
  --mode1-spatial-model flat \
  --out eval/results/efficientnet_b0_prefix9_dse.csv \
  --json-out eval/results/efficientnet_b0_prefix9_dse.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Layer = Dict[str, Any]


@dataclass(frozen=True)
class DseOptions:
    ptotal: int
    pv_max: int
    pf_max: int
    pc_max: int
    fixed_pc_mode2: Optional[int]
    fixed_pf_mode2: Optional[int]
    mode1_spatial_model: str
    groups_policy: str
    allow_pf_gt_f: bool
    allow_pv_gt_spatial: bool
    enforce_rtl_ready: bool
    allow_all_mode1: bool
    allow_all_mode2: bool
    require_mode2_suffix: bool
    verbose: bool


@dataclass
class Candidate:
    mode: int
    pv_m1: int
    pf_m1: int
    pc_m2: int
    pf_m2: int
    read_cycles: int
    compute_cycles: int
    cycles: int
    ops: int
    macs: int
    ce_compute: float
    ce_total: float


@dataclass
class Solution:
    transition_layer_0based: Optional[int]
    transition_layer_1based: Optional[int]
    total_cycles: int
    total_compute_cycles: int
    total_read_cycles: int
    total_ops: int
    total_macs: int
    rows: List[Candidate]
    warnings: List[str]


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"ceil_div divisor must be positive, got {b}")
    return (a + b - 1) // b


def as_int(x: Any, name: str) -> int:
    if x is None:
        raise KeyError(f"Missing required layer field: {name}")
    return int(x)


def factor_pairs(product: int) -> List[Tuple[int, int]]:
    """Return all positive integer pairs (a, b) with a*b=product."""
    if product <= 0:
        raise ValueError("Ptotal must be positive")
    pairs: List[Tuple[int, int]] = []
    root = int(math.isqrt(product))
    for a in range(1, root + 1):
        if product % a == 0:
            b = product // a
            pairs.append((a, b))
            if a != b:
                pairs.append((b, a))
    pairs.sort()
    return pairs


def get_layer_id(layer: Layer, fallback: int) -> int:
    return int(layer.get("id", fallback))


def get_layer_name(layer: Layer, fallback: int) -> str:
    return str(layer.get("name", f"layer{fallback}"))


def get_layer_dims(layer: Layer) -> Tuple[int, int, int, int, int, int, int]:
    """Return hi, wi, c, ho, wo, f, k."""
    hi = as_int(layer.get("h_in"), "h_in")
    wi = as_int(layer.get("w_in"), "w_in")
    c = as_int(layer.get("c_in"), "c_in")
    f = as_int(layer.get("f_out"), "f_out")
    k = as_int(layer.get("k"), "k")
    ho = as_int(layer.get("conv_h_out", layer.get("h_out")), "conv_h_out")
    wo = as_int(layer.get("conv_w_out", layer.get("w_out")), "conv_w_out")
    return hi, wi, c, ho, wo, f, k


def groups_for_layer(layer: Layer) -> int:
    return int(layer.get("groups", 1) or 1)


def effective_c_for_cycles(layer: Layer, opts: DseOptions) -> Tuple[int, List[str]]:
    """
    Return channel count used by the cycle model.

    groups_policy:
      - error:           reject groups != 1. This is the default for current CNN_V4 evaluation.
      - warn-dense:      always use dense C. Warn if groups != 1.
      - native-grouped:  use C/groups for grouped/depthwise theoretical model.

    Current CNN_V4 RTL is dense-conv oriented. The safest default is to
    reject grouped/depthwise layers unless the user explicitly chooses another policy.
    """
    _, _, c, _, _, f, _ = get_layer_dims(layer)
    groups = groups_for_layer(layer)
    warnings: List[str] = []

    if groups <= 1:
        return c, warnings

    lname = get_layer_name(layer, int(layer.get("id", -1)))

    if opts.groups_policy == "error":
        raise ValueError(
            f"Layer {lname} has groups={groups}. Current DSE run uses --groups-policy error."
        )

    if opts.groups_policy == "native-grouped":
        if c % groups != 0:
            raise ValueError(f"Layer {lname}: c_in={c} is not divisible by groups={groups}")
        return c // groups, warnings

    if opts.groups_policy == "warn-dense":
        warnings.append(
            f"Layer {lname}: groups={groups}; using dense-conv cycle model with C={c}. "
            "This is compatible with dense emulation but not native depthwise/grouped execution."
        )
        return c, warnings

    raise ValueError(f"Unknown groups policy: {opts.groups_policy}")


def layer_macs(layer: Layer, opts: DseOptions) -> Tuple[int, List[str]]:
    """Return useful MAC count for CE reporting."""
    _, _, c, ho, wo, f, k = get_layer_dims(layer)
    groups = groups_for_layer(layer)
    warnings: List[str] = []

    # If descriptor already has ops, use it for the benchmark's useful-work definition.
    # This matters for depthwise descriptors, whose useful ops are much smaller than dense ops.
    if "ops" in layer and layer["ops"] is not None:
        ops = int(layer["ops"])
        return ops // 2, warnings

    if groups <= 1 or opts.groups_policy == "warn-dense":
        return ho * wo * f * c * k * k, warnings

    if opts.groups_policy == "native-grouped":
        if c % groups != 0:
            raise ValueError(f"c_in={c} is not divisible by groups={groups}")
        return ho * wo * f * (c // groups) * k * k, warnings

    return ho * wo * f * c * k * k, warnings


def layer_ops(layer: Layer, opts: DseOptions) -> Tuple[int, List[str]]:
    macs, warnings = layer_macs(layer, opts)
    return 2 * macs, warnings


def mode1_read_cycles(layer: Layer, pv: int, opts: DseOptions) -> int:
    hi, wi, _, _, _, _, k = get_layer_dims(layer)

    # DCP-CNN paper: read startup in mode 1 roughly ceil(W/Pv) * K.
    # For current CNN_V4, IFM buffer stores Pv pixels per address along a row, so this is a
    # reasonable front-end read startup approximation.
    return ceil_div(wi, pv) * k


def mode2_read_cycles(layer: Layer, pc: int, opts: DseOptions) -> int:
    _, _, c, _, _, _, k = get_layer_dims(layer)
    c_eff, _ = effective_c_for_cycles(layer, opts)
    return ceil_div(c_eff, pc) * k


def mode1_compute_cycles(layer: Layer, pv: int, pf: int, opts: DseOptions) -> int:
    _, _, c, ho, wo, f, k = get_layer_dims(layer)
    c_eff, _ = effective_c_for_cycles(layer, opts)

    if opts.mode1_spatial_model == "row":
        spatial_groups = ho * ceil_div(wo, pv)
    elif opts.mode1_spatial_model == "flat":
        spatial_groups = ceil_div(ho * wo, pv)
    elif opts.mode1_spatial_model == "dcp-height":
        # Mirrors DCP-CNN Eq. (3): K^2 * Wo * C * ceil(Ho/Pv) * ceil(F/Pf).
        spatial_groups = wo * ceil_div(ho, pv)
    else:
        raise ValueError(f"Unknown mode1 spatial model: {opts.mode1_spatial_model}")

    return (k * k) * c_eff * spatial_groups * ceil_div(f, pf)


def mode2_compute_cycles(layer: Layer, pc: int, pf: int, opts: DseOptions) -> int:
    _, _, _, ho, wo, f, k = get_layer_dims(layer)
    c_eff, _ = effective_c_for_cycles(layer, opts)
    return (k * k) * ho * wo * ceil_div(c_eff, pc) * ceil_div(f, pf)


def make_candidate_mode1(layer: Layer, pv: int, pf: int, opts: DseOptions) -> Candidate:
    read = mode1_read_cycles(layer, pv, opts)
    comp = mode1_compute_cycles(layer, pv, pf, opts)
    cycles = read + comp
    ops, _ = layer_ops(layer, opts)
    macs = ops // 2
    ce_compute = macs / (opts.ptotal * comp) if comp > 0 else 0.0
    ce_total = macs / (opts.ptotal * cycles) if cycles > 0 else 0.0
    return Candidate(
        mode=1,
        pv_m1=pv,
        pf_m1=pf,
        pc_m2=0,
        pf_m2=0,
        read_cycles=read,
        compute_cycles=comp,
        cycles=cycles,
        ops=ops,
        macs=macs,
        ce_compute=ce_compute,
        ce_total=ce_total,
    )


def make_candidate_mode2(layer: Layer, pc: int, pf: int, opts: DseOptions) -> Candidate:
    read = mode2_read_cycles(layer, pc, opts)
    comp = mode2_compute_cycles(layer, pc, pf, opts)
    cycles = read + comp
    ops, _ = layer_ops(layer, opts)
    macs = ops // 2
    ce_compute = macs / (opts.ptotal * comp) if comp > 0 else 0.0
    ce_total = macs / (opts.ptotal * cycles) if cycles > 0 else 0.0
    return Candidate(
        mode=2,
        pv_m1=0,
        pf_m1=0,
        pc_m2=pc,
        pf_m2=pf,
        read_cycles=read,
        compute_cycles=comp,
        cycles=cycles,
        ops=ops,
        macs=macs,
        ce_compute=ce_compute,
        ce_total=ce_total,
    )


def valid_m1_pairs(layer: Layer, opts: DseOptions) -> List[Tuple[int, int]]:
    _, _, _, ho, wo, f, _ = get_layer_dims(layer)
    spatial = ho * wo
    pairs: List[Tuple[int, int]] = []

    for pv, pf in factor_pairs(opts.ptotal):
        if pv > opts.pv_max or pf > opts.pf_max:
            continue
        if (not opts.allow_pf_gt_f) and pf > f:
            continue
        if (not opts.allow_pv_gt_spatial) and pv > spatial:
            continue

        # For current CNN_V4 row-wise pixel grouping, Pv much larger than W can be valid only
        # under the flat model. Under row/dcp-height, keep it bounded by the active dimension.
        if opts.mode1_spatial_model == "row" and (not opts.allow_pv_gt_spatial) and pv > wo:
            continue
        if opts.mode1_spatial_model == "dcp-height" and (not opts.allow_pv_gt_spatial) and pv > ho:
            continue

        pairs.append((pv, pf))

    return pairs


def valid_m2_pairs_for_layers(layers: Sequence[Layer], opts: DseOptions) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []

    if opts.fixed_pc_mode2 is not None or opts.fixed_pf_mode2 is not None:
        if opts.fixed_pc_mode2 is None or opts.fixed_pf_mode2 is None:
            raise ValueError("Specify both --fixed-pc-mode2 and --fixed-pf-mode2, or neither.")
        if opts.fixed_pc_mode2 * opts.fixed_pf_mode2 != opts.ptotal:
            raise ValueError(
                f"fixed Pc/Pf product {opts.fixed_pc_mode2}*{opts.fixed_pf_mode2} "
                f"does not equal Ptotal={opts.ptotal}"
            )
        base_pairs = [(opts.fixed_pc_mode2, opts.fixed_pf_mode2)]
    elif opts.fixed_pc_mode2 is not None:
        if opts.ptotal % opts.fixed_pc_mode2 != 0:
            raise ValueError("Ptotal is not divisible by fixed Pc")
        base_pairs = [(opts.fixed_pc_mode2, opts.ptotal // opts.fixed_pc_mode2)]
    else:
        base_pairs = factor_pairs(opts.ptotal)

    for pc, pf in base_pairs:
        if pc > opts.pc_max or pf > opts.pf_max:
            continue

        valid = True
        for layer in layers:
            _, _, c, _, _, f, _ = get_layer_dims(layer)
            c_eff, _ = effective_c_for_cycles(layer, opts)

            if pc > c_eff:
                valid = False
                break
            if (not opts.allow_pf_gt_f) and pf > f:
                valid = False
                break

        if valid:
            pairs.append((pc, pf))

    return pairs


def best_mode1_for_layer(layer: Layer, opts: DseOptions) -> Optional[Candidate]:
    pairs = valid_m1_pairs(layer, opts)
    if not pairs:
        return None

    best: Optional[Candidate] = None
    for pv, pf in pairs:
        cand = make_candidate_mode1(layer, pv, pf, opts)
        if best is None or cand.cycles < best.cycles:
            best = cand
    return best


def best_mode2_for_suffix(layers: Sequence[Layer], opts: DseOptions) -> Optional[Tuple[List[Candidate], int, int]]:
    pairs = valid_m2_pairs_for_layers(layers, opts)
    if not pairs:
        return None

    best_rows: Optional[List[Candidate]] = None
    best_total: Optional[int] = None
    best_pair: Tuple[int, int] = (0, 0)

    for pc, pf in pairs:
        rows = [make_candidate_mode2(layer, pc, pf, opts) for layer in layers]
        total = sum(r.cycles for r in rows)
        if best_total is None or total < best_total:
            best_total = total
            best_rows = rows
            best_pair = (pc, pf)

    assert best_rows is not None and best_total is not None
    return best_rows, best_pair[0], best_pair[1]


def collect_descriptor_warnings(layers: Sequence[Layer], opts: DseOptions) -> List[str]:
    warnings: List[str] = []
    for i, layer in enumerate(layers):
        lname = get_layer_name(layer, i)

        if opts.enforce_rtl_ready and layer.get("cnn_v4_current_rtl_ready") is False:
            raise ValueError(
                f"Layer {lname} has cnn_v4_current_rtl_ready=false. "
                "Run without --enforce-rtl-ready to only warn."
            )

        if layer.get("cnn_v4_current_rtl_ready") is False:
            warnings.append(
                f"Layer {lname}: cnn_v4_current_rtl_ready=false; DSE result may require RTL/preprocessing support."
            )

        for w in layer.get("cnn_v4_current_rtl_warnings", []) or []:
            warnings.append(f"Layer {lname}: {w}")

        _, gwarnings = effective_c_for_cycles(layer, opts)
        warnings.extend(gwarnings)

    return warnings


def run_dse(model: Dict[str, Any], opts: DseOptions) -> Solution:
    layers: List[Layer] = list(model.get("layers", []))
    if not layers:
        raise ValueError("Descriptor has no layers")

    warnings = collect_descriptor_warnings(layers, opts)
    n = len(layers)
    best_solution: Optional[Solution] = None

    # L is 0-based index of first mode-2 layer.
    # L = 0       => all mode 2
    # L = n       => all mode 1
    # 0 < L < n   => layers[0:L] mode 1, layers[L:n] mode 2
    start_L = 0 if opts.allow_all_mode2 else 1
    end_L_exclusive = n + 1 if opts.allow_all_mode1 else n

    for L in range(start_L, end_L_exclusive):
        if opts.require_mode2_suffix and L == n:
            continue

        rows: List[Candidate] = []
        valid = True

        # Prefix in mode 1.
        for i in range(0, L):
            cand = best_mode1_for_layer(layers[i], opts)
            if cand is None:
                valid = False
                break
            rows.append(cand)

        if not valid:
            continue

        # Suffix in mode 2, if any.
        if L < n:
            suffix = best_mode2_for_suffix(layers[L:n], opts)
            if suffix is None:
                continue
            suffix_rows, _, _ = suffix
            rows.extend(suffix_rows)

        total_cycles = sum(r.cycles for r in rows)
        total_compute_cycles = sum(r.compute_cycles for r in rows)
        total_read_cycles = sum(r.read_cycles for r in rows)
        total_ops = sum(r.ops for r in rows)
        total_macs = total_ops // 2

        sol = Solution(
            transition_layer_0based=(L if L < n else None),
            transition_layer_1based=(L + 1 if L < n else None),
            total_cycles=total_cycles,
            total_compute_cycles=total_compute_cycles,
            total_read_cycles=total_read_cycles,
            total_ops=total_ops,
            total_macs=total_macs,
            rows=rows,
            warnings=warnings,
        )

        if best_solution is None or sol.total_cycles < best_solution.total_cycles:
            best_solution = sol

    if best_solution is None:
        raise RuntimeError(
            "No valid DSE solution found. Try smaller Ptotal, larger max limits, "
            "--allow-all-mode1, or --allow-pf-gt-f if your RTL supports partial filters."
        )

    return best_solution


def write_csv(model: Dict[str, Any], sol: Solution, path: Path, opts: DseOptions) -> None:
    layers: List[Layer] = list(model["layers"])
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name",
            "layer_id",
            "layer_name",
            "op_type",
            "groups",
            "h_in",
            "w_in",
            "c_in",
            "conv_h_out",
            "conv_w_out",
            "f_out",
            "k",
            "conv_stride",
            "pool_enable",
            "mode",
            "pv_m1",
            "pf_m1",
            "pc_m2",
            "pf_m2",
            "read_cycles_est",
            "compute_cycles_est",
            "cycles_est",
            "ops",
            "macs",
            "ce_compute_est",
            "ce_total_est",
            "rtl_ready",
        ])

        for i, (layer, cand) in enumerate(zip(layers, sol.rows)):
            hi, wi, c, ho, wo, fout, k = get_layer_dims(layer)
            pool = layer.get("pool") or {}
            writer.writerow([
                model.get("model_name", model.get("name", "model")),
                get_layer_id(layer, i),
                get_layer_name(layer, i),
                layer.get("op_type", "conv2d"),
                groups_for_layer(layer),
                hi,
                wi,
                c,
                ho,
                wo,
                fout,
                k,
                int(layer.get("conv_stride", 1) or 1),
                bool(pool.get("enable", False)),
                cand.mode,
                cand.pv_m1,
                cand.pf_m1,
                cand.pc_m2,
                cand.pf_m2,
                cand.read_cycles,
                cand.compute_cycles,
                cand.cycles,
                cand.ops,
                cand.macs,
                f"{cand.ce_compute:.6f}",
                f"{cand.ce_total:.6f}",
                layer.get("cnn_v4_current_rtl_ready", "unknown"),
            ])


def solution_summary_dict(model: Dict[str, Any], sol: Solution, opts: DseOptions) -> Dict[str, Any]:
    return {
        "model_name": model.get("model_name", model.get("name", "model")),
        "num_layers": len(model.get("layers", [])),
        "ptotal": opts.ptotal,
        "pv_max": opts.pv_max,
        "pf_max": opts.pf_max,
        "pc_max": opts.pc_max,
        "mode1_spatial_model": opts.mode1_spatial_model,
        "groups_policy": opts.groups_policy,
        "transition_layer_0based": sol.transition_layer_0based,
        "transition_layer_1based": sol.transition_layer_1based,
        "total_cycles_est": sol.total_cycles,
        "total_compute_cycles_est": sol.total_compute_cycles,
        "total_read_cycles_est": sol.total_read_cycles,
        "total_ops": sol.total_ops,
        "total_macs": sol.total_macs,
        "ce_total_est": sol.total_macs / (opts.ptotal * sol.total_cycles) if sol.total_cycles else 0.0,
        "ce_compute_est": sol.total_macs / (opts.ptotal * sol.total_compute_cycles) if sol.total_compute_cycles else 0.0,
        "warnings": sol.warnings,
    }


def write_summary_json(model: Dict[str, Any], sol: Solution, path: Path, opts: DseOptions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(solution_summary_dict(model, sol, opts), f, indent=2)


def write_updated_descriptor_json(model: Dict[str, Any], sol: Solution, path: Path, opts: DseOptions) -> None:
    updated = copy.deepcopy(model)
    updated["dse_lite"] = solution_summary_dict(model, sol, opts)

    for layer, cand in zip(updated["layers"], sol.rows):
        layer["dse"] = {
            "mode": cand.mode,
            "pv_m1": cand.pv_m1 if cand.mode == 1 else None,
            "pf_m1": cand.pf_m1 if cand.mode == 1 else None,
            "pc_m2": cand.pc_m2 if cand.mode == 2 else None,
            "pf_m2": cand.pf_m2 if cand.mode == 2 else None,
            "cycles_est": cand.cycles,
            "read_cycles_est": cand.read_cycles,
            "compute_cycles_est": cand.compute_cycles,
            "ce_compute_est": cand.ce_compute,
            "ce_total_est": cand.ce_total,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(updated, f, indent=2)


def print_summary(model: Dict[str, Any], sol: Solution, opts: DseOptions) -> None:
    name = model.get("model_name", model.get("name", "model"))
    print(f"Model                  : {name}")
    print(f"Layers                 : {len(model.get('layers', []))}")
    print(f"Ptotal                 : {opts.ptotal}")
    print(f"Mode1 spatial model    : {opts.mode1_spatial_model}")
    print(f"DSE script version     : v2_nomobilenet_safe_groups")
    print(f"Groups policy          : {opts.groups_policy}")
    print(f"Transition layer 0base : {sol.transition_layer_0based}")
    print(f"Transition layer 1base : {sol.transition_layer_1based}")
    print(f"Total cycles est.      : {sol.total_cycles}")
    print(f"Total compute cycles   : {sol.total_compute_cycles}")
    print(f"Total read cycles      : {sol.total_read_cycles}")
    print(f"Total GOPs             : {sol.total_ops / 1e9:.6f}")
    ce_compute = sol.total_macs / (opts.ptotal * sol.total_compute_cycles) if sol.total_compute_cycles else 0.0
    ce_total = sol.total_macs / (opts.ptotal * sol.total_cycles) if sol.total_cycles else 0.0
    print(f"CE compute est.        : {ce_compute * 100:.2f}%")
    print(f"CE total est.          : {ce_total * 100:.2f}%")

    if sol.warnings:
        print(f"Warnings               : {len(sol.warnings)}")
        for w in sol.warnings[:12]:
            print(f"  - {w}")
        if len(sol.warnings) > 12:
            print(f"  ... {len(sol.warnings) - 12} more warnings")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DSE-lite schedule generator for CNN_V4 benchmark layer descriptors."
    )
    parser.add_argument("--model", required=True, help="Input layer descriptor JSON")
    parser.add_argument("--ptotal", type=int, required=True, help="Total MAC parallelism")
    parser.add_argument("--pv-max", type=int, default=128, help="Max/runtime width for Pv in mode 1")
    parser.add_argument("--pf-max", type=int, default=128, help="Max/runtime width for Pf")
    parser.add_argument("--pc-max", type=int, default=32, help="Max/runtime width for Pc in mode 2")
    parser.add_argument("--fixed-pc-mode2", type=int, default=None, help="Force Pc in mode 2")
    parser.add_argument("--fixed-pf-mode2", type=int, default=None, help="Force Pf in mode 2")
    parser.add_argument(
        "--mode1-spatial-model",
        choices=["row", "flat", "dcp-height"],
        default="flat",
        help=(
            "Mode-1 pixel parallelism model. row: H*ceil(W/Pv), best for strict row-wise IFM buffer; "
            "flat: ceil(H*W/Pv); dcp-height: W*ceil(H/Pv), matching DCP-CNN equation style."
        ),
    )
    parser.add_argument(
        "--groups-policy",
        choices=["warn-dense", "native-grouped", "error"],
        default="error",
        help=(
            "How to treat grouped/depthwise layers. error rejects groups!=1 and is the safe default for current CNN_V4; "
            "warn-dense uses dense C cycles and warns; native-grouped uses C/groups theoretical cycles for exploration only."
        ),
    )
    parser.add_argument("--allow-pf-gt-f", action="store_true", help="Allow Pf > F for partial-filter execution")
    parser.add_argument("--allow-pv-gt-spatial", action="store_true", help="Allow Pv greater than active spatial dimension")
    parser.add_argument("--enforce-rtl-ready", action="store_true", help="Fail if descriptor marks a layer not ready for current CNN_V4 RTL")
    parser.add_argument("--allow-all-mode1", action="store_true", default=True, help="Allow solution with no mode-2 suffix")
    parser.add_argument("--no-all-mode1", dest="allow_all_mode1", action="store_false", help="Forbid all-mode1 solution")
    parser.add_argument("--allow-all-mode2", action="store_true", help="Allow solution starting directly in mode 2")
    parser.add_argument("--require-mode2-suffix", action="store_true", help="Require at least one layer to use mode 2")
    parser.add_argument("--out", required=True, help="Output CSV schedule")
    parser.add_argument("--json-out", default=None, help="Optional updated descriptor JSON with dse fields filled")
    parser.add_argument("--summary-json", default=None, help="Optional compact summary JSON")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    model_path = Path(args.model)
    with model_path.open("r") as f:
        model = json.load(f)

    opts = DseOptions(
        ptotal=args.ptotal,
        pv_max=args.pv_max,
        pf_max=args.pf_max,
        pc_max=args.pc_max,
        fixed_pc_mode2=args.fixed_pc_mode2,
        fixed_pf_mode2=args.fixed_pf_mode2,
        mode1_spatial_model=args.mode1_spatial_model,
        groups_policy=args.groups_policy,
        allow_pf_gt_f=args.allow_pf_gt_f,
        allow_pv_gt_spatial=args.allow_pv_gt_spatial,
        enforce_rtl_ready=args.enforce_rtl_ready,
        allow_all_mode1=args.allow_all_mode1,
        allow_all_mode2=args.allow_all_mode2,
        require_mode2_suffix=args.require_mode2_suffix,
        verbose=args.verbose,
    )

    sol = run_dse(model, opts)

    csv_path = Path(args.out)
    write_csv(model, sol, csv_path, opts)

    if args.json_out:
        write_updated_descriptor_json(model, sol, Path(args.json_out), opts)

    if args.summary_json:
        write_summary_json(model, sol, Path(args.summary_json), opts)

    print_summary(model, sol, opts)
    print(f"CSV schedule written   : {csv_path}")
    if args.json_out:
        print(f"Updated JSON written   : {args.json_out}")
    if args.summary_json:
        print(f"Summary JSON written   : {args.summary_json}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
