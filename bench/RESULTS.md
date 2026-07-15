# BLIS `gb10` config — performance on NVIDIA GB10 (DGX Spark), 10× Cortex-X925

All numbers GFLOP/s, pinned to the 10 X925 P-cores (`taskset -c 5-9,15-19`,
`OMP_PROC_BIND=close OMP_PLACES=cores`). Compared against NVIDIA NVPL and OpenBLAS.

## Hardware / roofline
- 10× Cortex-X925 @3.9 GHz, 6 FP pipes (128-bit NEON), L1d 64K, L2 2M, 2× L3 domain.
- FMA roofline: 89.5/core, **898 (DP) / 1787 (SP)** across 10 cores.
- Sustained DRAM BW ~118 GB/s (STREAM). GEMM is *not* DRAM-bound; the multicore
  DGEMM ceiling (~88%) is per-cluster L3 bandwidth + the 128-bit vector width's
  arithmetic intensity (verified: identical per-core cache/stall counts 1→10 cores).

## Performance sweep (10× X925, GFLOP/s, default auto-threading)
Full size sweep vs NVPL / OpenBLAS. "win" = fastest of the three.

DGEMM square (peak 898): BLIS wins ≥256; NVPL/OpenBLAS win the tiny end.
```
 size    BLIS   %pk    NVPL  OpenBLAS  win
   32    33.3   3.7%   18.6    53.2   openblas
   64    55.4   6.2%   88.6    60.5   nvpl
  128   232.2  25.9%  241.2   182.6   nvpl
  256   459.9  51.2%  451.3   236.0   BLIS
  512   633.9  70.6%  602.4   400.1   BLIS
 1024   711.9  79.3%  695.9   392.8   BLIS
 2048   719.8  80.2%  697.6   499.4   BLIS
 4096   746.0  83.1%  714.6   550.1   BLIS
 8192   753.8  83.9%  716.7   541.2   BLIS
```
SGEMM square (peak 1787): BLIS wins ≥1024; small SGEMM improved by the
precision-aware threading threshold (64³ 35→86).
```
 size    BLIS   %pk    NVPL  OpenBLAS  win
   64    86.2   4.8%  117.5   118.7   openblas
  256   681.3  38.1%  869.8   515.0   nvpl
  512  1318.1  73.8% 1383.3   739.8   nvpl
 1024  1531.4  85.7% 1509.7   559.0   BLIS
 4096  1584.8  88.7% 1551.5  1151.1   BLIS
 8192  1611.4  90.2% 1580.8  1165.3   BLIS
```
DGEMM rectangular — **BLIS wins every case**:
```
 rank-k 4096²×K:  K=8→135, 16→270, 32→488, 64→669, 128→715, 256→756, 512→748  (all > NVPL)
 tall  M×256×256: 1k→652, 4k→672, 16k→734, 64k→710
 wide  256×N×256: 1k→639, 4k→631, 16k→667, 64k→666
 deep  256²×K:    1k→513, 4k→521, 16k→489
```
(Forced BLIS_JC_NT/IC_NT gives only ~1.5% over auto and is size-dependent, so
auto is kept as the default.)

## GEMM — square (DGEMM)
| size | BLIS | NVPL | OpenBLAS |
|---|---|---|---|
| 512 | **672** | 613 | 345 |
| 1024 | **714** | 695 | 414 |
| 2048 | **722** | 702 | 502 |
| 4000 | **771** | 725 | 574 |
| 8000 | **787 (87.7%)** | 721 | 573 |

## GEMM — rank-k (M=N=4000, small K) — BLIS wins all after AND-routing fix
| K | BLIS | NVPL | OpenBLAS |
|---|---|---|---|
| 16 | **266** | 236 | 114 |
| 32 | **472** | 418 | 211 |
| 64 | **667** | 590 | 342 |
| 128 | **736** | 688 | 466 |
| 256 | **772** | 615 | 519 |

## GEMM — small square (SUP path, single-thread below m*n*k=200k)
| size | BLIS | NVPL | OpenBLAS | note |
|---|---|---|---|---|
| 16³ | 11 | 21 | **34** | tiny: OpenBLAS's dedicated small kernels win |
| 64³ | 55 | **87** | 60 | |
| 128³ | 228 | **239** | 199 | |
| 192³ | **391** | 383 | 199 | |
| 384³ | 590 | **598** | 211 | |

Small-matrix threading cutoff fixed a catastrophe (16³ was 1.1 with 10 threads → 11
single-threaded). Remaining sub-96 gap vs OpenBLAS is micro-kernel quality (would
need hand-written tiny-GEMM kernels).

## Non-GEMM — level-3 (BLIS wins; inherits GEMM microkernel)
| op (4000) | BLIS | NVPL | OpenBLAS |
|---|---|---|---|
| syrk | **732** | 685 | 541 |
| trsm | **723** | 663 | 579 |
| trmm | **742** | 683 | 581 |

## Non-GEMM — level-1/2 (now multithreaded for gb10; were single-threaded)
BLIS runs level-1/2 single-threaded by default (so does NVPL). gb10 threads them
via `BLIS_ENABLE_L1_OPENMP` (axpyv/dotv/scal/copy/asumv/normfv + gemv + ger).
| op | before | BLIS | NVPL | OpenBLAS |
|---|---|---|---|---|
| gemv 8000² (GFLOP/s) | 8.5 | **22** | 22 | 19 |
| ger 8000² (GFLOP/s) | 7.2 | **17** | — | — |
| axpy 64M (GFLOP/s) | 4.1 | **11.0** | 4.1 | 11.3 |
| dot 64M (GFLOP/s) | 4.3 | **12.1** | 4.3 | 12.6 |
| asum 64M (GB/s) | 15 | **90** | — | — |
| nrm2 64M (GB/s) | 7.3 | **69** | — | — |
nrm2 threaded while preserving overflow-safe scaling (verified: 1e200 vector →
1e203, no overflow). BLIS-gb10 is now ~3–10× NVPL on level-1/2 (NVPL doesn't thread them).

## Tuning summary (all in the `gb10` config / gated by config macros)
- NEON 6×8 dgemm / 8×12 sgemm asm kernels, `-mcpu=gb10`, MC=KC=336, NC=4080.
- SUP path wired (armv8a gemmsup) with **AND** threshold routing
  (`BLIS_SUP_THRESH_ALL`): SUP only when all dims small; skinny/rank-k → native.
- Small-matrix single-thread cutoff (`BLIS_SMALL_MT_THRESHOLD=200000`).
- Huge-page memory pool (`BLIS_ENABLE_HUGEPAGE_POOL`, 1G→2M→malloc).
- Multithreaded level-1v (axpyv/dotv/scal2v/copyv/scalv…) and level-2 gemv
  (`BLIS_ENABLE_L1_OPENMP`).

## Remaining weak spots (diminishing returns)
- Very small square (≤48) and extreme-skinny (dim ≤ 32): OpenBLAS's hand-tuned
  small kernels win; BLIS generic SUP trails. Absolute times are tiny.
- Multicore DGEMM square ~88% of FMA roofline (vector-width/L3 bound, beats NVPL 80%).
