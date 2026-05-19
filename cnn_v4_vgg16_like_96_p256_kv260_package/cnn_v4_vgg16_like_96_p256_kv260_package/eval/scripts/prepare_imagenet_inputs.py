import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def pack_u8_to_u32_le(values):
    """
    Pack 4 uint8 values into one 32-bit word, little-endian:
      word[7:0]   = byte0
      word[15:8]  = byte1
      word[23:16] = byte2
      word[31:24] = byte3
    """
    vals = [int(v) & 0xFF for v in values]
    words = []

    for i in range(0, len(vals), 4):
        b0 = vals[i + 0] if i + 0 < len(vals) else 0
        b1 = vals[i + 1] if i + 1 < len(vals) else 0
        b2 = vals[i + 2] if i + 2 < len(vals) else 0
        b3 = vals[i + 3] if i + 3 < len(vals) else 0

        word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        words.append(word)

    return words


def write_u32_hex(path, words):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for word in words:
            f.write(f"{word & 0xFFFFFFFF:08X}\n")


def load_image_paths(imagenet_val_dir):
    exts = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG"}
    paths = []

    for p in Path(imagenet_val_dir).iterdir():
        if p.suffix in exts:
            paths.append(p)

    return sorted(paths)


def preprocess_image_to_uint8_hwc(path, height, width):
    """
    Convert ImageNet JPEG to RGB uint8 tensor [H, W, C].

    Important:
      - No mean/std normalization here.
      - This is raw uint8 image data for FPGA input.
      - ImageOps.fit performs resize + center crop to exact size.
    """
    img = Image.open(path).convert("RGB")

    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR

    img = ImageOps.fit(
        img,
        size=(width, height),
        method=resample,
        centering=(0.5, 0.5),
    )

    arr = np.array(img, dtype=np.uint8)

    if arr.shape != (height, width, 3):
        raise RuntimeError(f"Unexpected shape {arr.shape}, expected {(height, width, 3)}")

    return arr, img


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--imagenet-val-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)

    args = parser.parse_args()

    image_paths = load_image_paths(args.imagenet_val_dir)

    if not image_paths:
        raise RuntimeError(f"No images found in {args.imagenet_val_dir}")

    selected = image_paths[args.start_index: args.start_index + args.count]

    if len(selected) < args.count:
        raise RuntimeError(
            f"Requested {args.count} images, but only found {len(selected)} from start index {args.start_index}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": "ImageNet ILSVRC2012 validation images",
        "preprocess": "RGB + resize/center-crop to target HxW + raw uint8 HWC",
        "normalization": "none",
        "tensor_layout": "HWC",
        "hex_pack": "HWC flatten, 4 uint8 per 32-bit word, little-endian",
        "height": args.height,
        "width": args.width,
        "channels": 3,
        "images": []
    }

    for idx, img_path in enumerate(selected):
        image_id = f"img{idx:04d}"

        arr, processed_img = preprocess_image_to_uint8_hwc(
            img_path,
            height=args.height,
            width=args.width,
        )

        image_dir = out_dir / image_id
        image_dir.mkdir(parents=True, exist_ok=True)

        npy_path = image_dir / "input_uint8_hwc.npy"
        hex_path = image_dir / "input_uint8_hwc_u32le.hex"
        png_path = image_dir / "input_preview.png"

        np.save(npy_path, arr)
        processed_img.save(png_path)

        flat_hwc = arr.reshape(-1)
        words = pack_u8_to_u32_le(flat_hwc)
        write_u32_hex(hex_path, words)

        metadata["images"].append({
            "image_id": image_id,
            "original_file": str(img_path),
            "npy": str(npy_path),
            "hex": str(hex_path),
            "preview_png": str(png_path),
            "shape_hwc": list(arr.shape),
            "num_pixels": int(args.height * args.width),
            "num_bytes": int(arr.size),
            "num_u32_words": int(len(words)),
        })

        print(f"[OK] {image_id}: {img_path.name} -> {hex_path}")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Output folder: {out_dir}")


if __name__ == "__main__":
    main()