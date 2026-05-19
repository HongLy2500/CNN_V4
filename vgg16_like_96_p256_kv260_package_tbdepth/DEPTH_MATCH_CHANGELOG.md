# Depth/compact-storage changelog

This revision updates the KV260 VGG16-like 96x96 P256 package to match the passing 96x96 simulation profile supplied by the user.

Changed parameters:

```text
HT              = 4
OFM_ROW_STRIDE  = 6
OFM_BANK_DEPTH  = H_MAX * OFM_ROW_STRIDE = 96 * 6 = 576
OFM_LINEAR_DEPTH= F_MAX * OFM_BANK_DEPTH = 128 * 576 = 73728
WGT_DEPTH       = 1024
```

The layer count remains 13. Mode/padding/Pv/Pf/PC settings are unchanged:

```text
Mode1: L0-L6, Pv=16, Pf=16
Mode2: L7-L12, PC=16, Pf=16
K=3, stride=1, padding=1
```

Rationale:
- The compact OFM row stride is 6, not 16, because the maximum physical row word count is 6 for this workload.
- `WGT_DEPTH=1024` is sufficient for the internal wide-weight buffer depth of this benchmark, matching the passing simulation profile.
