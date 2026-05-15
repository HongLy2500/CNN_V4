#!/usr/bin/env python3
"""
Compare CNN_V4 FPGA OFM readback against a golden OFM hex file.

Typical use
-----------
python eval/scripts/compare_ofm.py \
  --expected eval/golden/vgg16_conv13/img0000/expected_final_ofm_linear_u32le.hex \
  --actual   eval/results/vgg16_conv13/img0000/fpga_ofm_readback.hex \
  --out-report eval/results/vgg16_conv13/img0000/compare_report.txt \
  --summary-json eval/results/vgg16_conv13/img0000/compare_summary.json

Default comparison
------------------
- Hex files are interpreted as one 32-bit word per line.
- Lines may be either plain hex values, e.g. "000000FF", or simple log lines
  containing a data word, e.g. "OFM[0] @0x10000 = 0x000000FF".
- Comparison is word-by-word by default.
- Use --compare-mode byte to compare unpacked bytes instead.

Important
---------
This script compares the two files exactly as provided. If CNN_V4 writes OFM
in a tiled/banked/Pv/Pf/Pc-specific DDR layout, pack the golden OFM into the
same layout before running this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


HEX_TOKEN_RE = re.compile(r"0[xX][0-9a-fA-F]+|(?<![A-Za-z0-9_])[0-9a-fA-F]{1,16}(?![A-Za-z0-9_])")
DATA_FIELD_RE = re.compile(r"\b(?:data|word|value|val|read|rd)\s*[=:]\s*(0[xX][0-9a-fA-F]+|[0-9a-fA-F]{1,16})", re.IGNORECASE)


def parse_hex_token(token: str) -> int:
    token = token.strip()
    if token.lower().startswith("0x"):
        return int(token, 16)
    return int(token, 16)


@dataclass
class Mismatch:
    index: int
    expected: int
    actual: int
    xor: int


@dataclass
class CompareSummary:
    status: str
    compare_mode: str
    expected_file: str
    actual_file: str
    expected_items: int
    actual_items: int
    compared_items: int
    mismatch_count: int
    length_mismatch: bool
    mask: Optional[str]
    expected_offset: int
    actual_offset: int
    requested_count: Optional[int]
    report_file: Optional[str]
    generated_at: str


def parse_int_auto(text: str) -> int:
    """Parse decimal or hex integer from CLI."""
    return int(str(text), 0)


def strip_inline_comments(line: str) -> str:
    """Remove common inline comments without breaking 0x tokens."""
    # Treat # as a comment marker.
    if "#" in line:
        line = line.split("#", 1)[0]

    # Treat // as comment marker only when it is not part of a path-like string.
    # For normal hex files this is sufficient.
    if "//" in line:
        line = line.split("//", 1)[0]

    return line.strip()


def extract_hex_word_from_line(line: str, line_no: int) -> Optional[int]:
    """
    Extract one data word from a line.

    Supported examples:
      000000FF
      0x000000FF
      OFM[0] @0x10000 = 0x000000FF
      addr=0x10000 data=0x000000FF be=0xF
      0x10000: 0x000000FF

    Returns None for blank/comment-only lines.
    """
    original = line.rstrip("\n")
    line = strip_inline_comments(original)
    if not line:
        return None

    # Prefer explicit data-like fields, so a line with "data=... be=..." does
    # not accidentally parse byte-enable as the data word.
    m = DATA_FIELD_RE.search(line)
    if m:
        return parse_hex_token(m.group(1)) & 0xFFFFFFFF

    # If the line has an assignment, parse the first hex token after the last '='.
    if "=" in line:
        rhs = line.rsplit("=", 1)[1]
        tokens = HEX_TOKEN_RE.findall(rhs)
        if tokens:
            return parse_hex_token(tokens[0]) & 0xFFFFFFFF

    # If the line has an address separator, parse the first token after the last ':'.
    if ":" in line:
        rhs = line.rsplit(":", 1)[1]
        tokens = HEX_TOKEN_RE.findall(rhs)
        if tokens:
            return parse_hex_token(tokens[0]) & 0xFFFFFFFF

    tokens = HEX_TOKEN_RE.findall(line)
    if not tokens:
        raise ValueError(f"Line {line_no}: no hex value found: {original!r}")

    # Plain hex line -> one token. Log line without data/=/colon -> last token is
    # usually the value, not the index/address.
    return parse_hex_token(tokens[-1]) & 0xFFFFFFFF


def read_hex_words(path: Path) -> List[int]:
    words: List[int] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            value = extract_hex_word_from_line(line, line_no)
            if value is not None:
                words.append(value)

    return words


def slice_items(values: List[int], offset: int, count: Optional[int], label: str) -> List[int]:
    if offset < 0:
        raise ValueError(f"{label} offset must be non-negative, got {offset}")
    if offset > len(values):
        raise ValueError(f"{label} offset {offset} is beyond length {len(values)}")

    end = len(values) if count is None else offset + count
    if count is not None and count < 0:
        raise ValueError(f"{label} count must be non-negative, got {count}")
    if end > len(values):
        raise ValueError(
            f"{label} requested range offset={offset}, count={count} exceeds length {len(values)}"
        )

    return values[offset:end]


def words_to_bytes(words: Iterable[int], endian: str) -> List[int]:
    out: List[int] = []
    for word in words:
        word = int(word) & 0xFFFFFFFF
        if endian == "little":
            out.extend([
                word & 0xFF,
                (word >> 8) & 0xFF,
                (word >> 16) & 0xFF,
                (word >> 24) & 0xFF,
            ])
        elif endian == "big":
            out.extend([
                (word >> 24) & 0xFF,
                (word >> 16) & 0xFF,
                (word >> 8) & 0xFF,
                word & 0xFF,
            ])
        else:
            raise ValueError(f"Unsupported endian: {endian}")
    return out


def compare_lists(
    expected: List[int],
    actual: List[int],
    mask: Optional[int],
    max_mismatches_to_store: int,
) -> Tuple[int, List[Mismatch]]:
    n = min(len(expected), len(actual))
    mismatch_count = 0
    mismatches: List[Mismatch] = []

    for idx in range(n):
        e_raw = int(expected[idx])
        a_raw = int(actual[idx])
        e = e_raw if mask is None else (e_raw & mask)
        a = a_raw if mask is None else (a_raw & mask)
        if e != a:
            mismatch_count += 1
            if len(mismatches) < max_mismatches_to_store:
                mismatches.append(Mismatch(index=idx, expected=e_raw, actual=a_raw, xor=(e ^ a)))

    return mismatch_count, mismatches


def format_item(value: int, mode: str) -> str:
    if mode == "byte":
        return f"0x{value & 0xFF:02X}"
    return f"0x{value & 0xFFFFFFFF:08X}"


def make_report_text(summary: CompareSummary, mismatches: List[Mismatch]) -> str:
    lines: List[str] = []
    lines.append("CNN_V4 OFM Compare Report")
    lines.append("=" * 72)
    lines.append(f"Status           : {summary.status}")
    lines.append(f"Compare mode     : {summary.compare_mode}")
    lines.append(f"Expected file    : {summary.expected_file}")
    lines.append(f"Actual file      : {summary.actual_file}")
    lines.append(f"Expected items   : {summary.expected_items}")
    lines.append(f"Actual items     : {summary.actual_items}")
    lines.append(f"Compared items   : {summary.compared_items}")
    lines.append(f"Mismatch count   : {summary.mismatch_count}")
    lines.append(f"Length mismatch  : {summary.length_mismatch}")
    lines.append(f"Mask             : {summary.mask if summary.mask is not None else 'none'}")
    lines.append(f"Expected offset  : {summary.expected_offset}")
    lines.append(f"Actual offset    : {summary.actual_offset}")
    lines.append(f"Requested count  : {summary.requested_count if summary.requested_count is not None else 'all'}")
    lines.append(f"Generated at     : {summary.generated_at}")
    lines.append("")

    if mismatches:
        lines.append("First mismatches")
        lines.append("-" * 72)
        lines.append(f"{'index':>12}  {'expected':>12}  {'actual':>12}  {'xor':>12}")
        for mm in mismatches:
            lines.append(
                f"{mm.index:12d}  "
                f"{format_item(mm.expected, summary.compare_mode):>12}  "
                f"{format_item(mm.actual, summary.compare_mode):>12}  "
                f"{format_item(mm.xor, summary.compare_mode):>12}"
            )
        if summary.mismatch_count > len(mismatches):
            lines.append(f"... {summary.mismatch_count - len(mismatches)} more mismatches not shown")
    else:
        lines.append("No mismatches in compared range.")

    lines.append("")
    return "\n".join(lines)


def write_summary_csv(path: Path, summary: CompareSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row: Dict[str, object] = asdict(summary)
    header = list(row.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare FPGA OFM readback with golden OFM hex.")
    parser.add_argument("--expected", required=True, help="Golden/expected .hex file.")
    parser.add_argument("--actual", required=True, help="FPGA readback .hex/log file.")
    parser.add_argument(
        "--compare-mode",
        choices=["word", "byte"],
        default="word",
        help="Compare 32-bit words or unpacked bytes. Default: word.",
    )
    parser.add_argument(
        "--byte-endian",
        choices=["little", "big"],
        default="little",
        help="Endian used when --compare-mode byte unpacks 32-bit words. Default: little.",
    )
    parser.add_argument(
        "--expected-offset",
        type=int,
        default=0,
        help="Offset in comparison items. Words for word mode, bytes for byte mode.",
    )
    parser.add_argument(
        "--actual-offset",
        type=int,
        default=0,
        help="Offset in comparison items. Words for word mode, bytes for byte mode.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of items to compare. Words for word mode, bytes for byte mode. Default: all available.",
    )
    parser.add_argument(
        "--mask",
        type=parse_int_auto,
        default=None,
        help="Optional mask applied before compare, e.g. 0xFF or 0xFFFFFFFF.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=50,
        help="Maximum mismatch rows to print/write in report. Default: 50.",
    )
    parser.add_argument(
        "--allow-length-mismatch",
        action="store_true",
        help="Do not fail only because expected/actual lengths differ; compare the common range.",
    )
    parser.add_argument("--out-report", default=None, help="Optional text report path.")
    parser.add_argument("--summary-json", default=None, help="Optional JSON summary path.")
    parser.add_argument("--summary-csv", default=None, help="Optional CSV summary path.")
    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="Always exit with code 0 even if compare fails. Useful in exploratory debug.",
    )

    args = parser.parse_args()

    expected_path = Path(args.expected)
    actual_path = Path(args.actual)

    if not expected_path.exists():
        raise SystemExit(f"ERROR: expected file not found: {expected_path}")
    if not actual_path.exists():
        raise SystemExit(f"ERROR: actual file not found: {actual_path}")

    expected_words = read_hex_words(expected_path)
    actual_words = read_hex_words(actual_path)

    if args.compare_mode == "byte":
        expected_items = words_to_bytes(expected_words, args.byte_endian)
        actual_items = words_to_bytes(actual_words, args.byte_endian)
        default_mask = 0xFF
    else:
        expected_items = expected_words
        actual_items = actual_words
        default_mask = 0xFFFFFFFF

    mask = args.mask
    if mask is not None:
        mask &= default_mask

    expected_cmp = slice_items(expected_items, args.expected_offset, args.count, "expected")
    actual_cmp = slice_items(actual_items, args.actual_offset, args.count, "actual")

    mismatch_count, mismatches = compare_lists(
        expected=expected_cmp,
        actual=actual_cmp,
        mask=mask,
        max_mismatches_to_store=max(0, args.max_mismatches),
    )

    length_mismatch = len(expected_cmp) != len(actual_cmp)
    compared_items = min(len(expected_cmp), len(actual_cmp))

    passed = mismatch_count == 0 and (not length_mismatch or args.allow_length_mismatch)
    status = "PASS" if passed else "FAIL"

    summary = CompareSummary(
        status=status,
        compare_mode=args.compare_mode,
        expected_file=str(expected_path),
        actual_file=str(actual_path),
        expected_items=len(expected_cmp),
        actual_items=len(actual_cmp),
        compared_items=compared_items,
        mismatch_count=mismatch_count,
        length_mismatch=length_mismatch,
        mask=f"0x{mask:X}" if mask is not None else None,
        expected_offset=args.expected_offset,
        actual_offset=args.actual_offset,
        requested_count=args.count,
        report_file=args.out_report,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )

    report_text = make_report_text(summary, mismatches)
    print(report_text)

    if args.out_report:
        report_path = Path(args.out_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")

    if args.summary_json:
        summary_json_path = Path(args.summary_json)
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": asdict(summary),
                    "first_mismatches": [asdict(mm) for mm in mismatches],
                },
                f,
                indent=2,
            )

    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), summary)

    if passed or args.no_fail_exit:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
