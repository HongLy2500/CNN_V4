# gen_expected_final_ofm.py
# Generate expected_final_ofm.hex for:
# tb_cnn_top_9layer_m1_dcp_efficientnet_b0_tablevi_fullscale_nogolden.sv

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DATA_W = 8
PV_MAX = 128
WORD_HEX_DIGITS = PV_MAX * 2  # 128 lanes * 8-bit = 1024-bit DDR word

layers = [
    # h_in, w_in, c_in, f_out, k, pool_en
    (224, 224, 3,   32, 3, True),
    (111, 111, 32,  16, 3, False),
    (109, 109, 16,  24, 3, True),
    (53,  53,  24,  24, 3, False),
    (51,  51,  24,  40, 3, True),
    (24,  24,  40,  40, 3, False),
    (22,  22,  40,  80, 3, True),
    (10,  10,  80,  80, 3, False),
    (8,   8,   80, 192, 3, False),
]

def gen_l0_ifm(c: int, r: int, x: int) -> np.int8:
    return np.int8(((c * 7 + r * 3 + x * 5 + (r * x) % 11) % 7) - 3)

def gen_wgt(layer: int, f: int, c: int, ky: int, kx: int) -> np.int8:
    v = f * (7 + 2 * layer) + c * (3 + 2 * layer) + ky * (5 + layer) + kx * (11 + layer) + (1 + layer)
    return np.int8((v % 3) - 1)

def relu_sat_i8(x: np.ndarray) -> np.ndarray:
    # RTL Mode1: ReLU first, then saturate positive result to signed 8-bit max 127.
    y = np.maximum(x, 0)
    y = np.minimum(y, 127)
    return y.astype(np.int8)

def conv3x3_valid(ifm: np.ndarray, layer_idx: int, f_out: int) -> np.ndarray:
    # ifm shape: H, W, C
    h, w, c = ifm.shape
    k = 3

    wgt = np.empty((f_out, c, k, k), dtype=np.int8)
    for f in range(f_out):
        for ch in range(c):
            for ky in range(k):
                for kx in range(k):
                    wgt[f, ch, ky, kx] = gen_wgt(layer_idx, f, ch, ky, kx)

    # windows shape: Hout, Wout, C, K, K
    windows = sliding_window_view(ifm, (k, k), axis=(0, 1))
    # sliding_window_view gives Hout,Wout,C,K,K for this input layout.
    windows_i32 = windows.astype(np.int32)
    wgt_i32 = wgt.astype(np.int32)

    # out shape: Hout, Wout, F
    out = np.einsum("hwcky,fcky->hwf", windows_i32, wgt_i32, optimize=True)
    return relu_sat_i8(out)

def maxpool2x2_stride2(x: np.ndarray) -> np.ndarray:
    h, w, c = x.shape
    h2 = h // 2
    w2 = w // 2
    x_crop = x[:h2 * 2, :w2 * 2, :]
    y = x_crop.reshape(h2, 2, w2, 2, c).max(axis=(1, 3))
    return y.astype(np.int8)

def pack_i8_lane0_to_1024b_hex(v: np.int8) -> str:
    # Current final DDR read in this test uses FINAL_STORE_PACK=1,
    # so only lane 0 is meaningful; all other lanes are zero.
    b = int(np.uint8(v))
    return f"{b:0{WORD_HEX_DIGITS}x}"

def main():
    h0, w0, c0, _, _, _ = layers[0]

    ifm = np.empty((h0, w0, c0), dtype=np.int8)
    for r in range(h0):
        for x in range(w0):
            for c in range(c0):
                ifm[r, x, c] = gen_l0_ifm(c, r, x)

    act = ifm

    for li, (h_in, w_in, c_in, f_out, k, pool_en) in enumerate(layers):
        assert act.shape == (h_in, w_in, c_in), f"L{li}: got {act.shape}, expected {(h_in, w_in, c_in)}"
        conv = conv3x3_valid(act, li, f_out)
        act = maxpool2x2_stride2(conv) if pool_en else conv
        print(f"L{li}: output {act.shape}")

    # Final expected OFM logical order must match ofm_buffer DMA linear read:
    # channel -> row -> group/col. For FINAL_STORE_PACK=1, group == col.
    final = act
    h, w, f = final.shape
    assert (h, w, f) == (6, 6, 192), f"Unexpected final shape {final.shape}"

    with open("expected_final_ofm.hex", "w", encoding="utf-8") as fp:
        for ch in range(f):
            for row in range(h):
                for col in range(w):
                    fp.write(pack_i8_lane0_to_1024b_hex(final[row, col, ch]) + "\n")

    print(f"Wrote expected_final_ofm.hex with {f*h*w} DDR words")

if __name__ == "__main__":
    main()