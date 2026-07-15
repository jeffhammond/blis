# BLIS `gb10` config — performance on NVIDIA GB10 (DGX Spark), 10× Cortex-X925

All numbers GFLOP/s, pinned to the 10 X925 P-cores (`taskset -c 5-9,15-19`,
`OMP_PROC_BIND=close OMP_PLACES=cores`). Compared against NVIDIA NVPL and OpenBLAS.

## Hardware / roofline
- 10× Cortex-X925 @3.9 GHz, 6 FP pipes (128-bit NEON), L1d 64K, L2 2M, 2× L3 domain.
- FMA roofline: 89.5/core, **898 (DP) / 1787 (SP)** across 10 cores.
- Sustained DRAM BW ~118 GB/s (STREAM). GEMM is *not* DRAM-bound; the multicore
  DGEMM ceiling (~88%) is per-cluster L3 bandwidth + the 128-bit vector width's
  arithmetic intensity (verified: identical per-core cache/stall counts 1→10 cores).

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

## Non-GEMM — level-1/2 (now multithreaded for gb10; were single-threaded → 3–4× slower)
| op | BLIS | NVPL | OpenBLAS |
|---|---|---|---|
| gemv 4000² | 22 | **23** | 18 |
| gemv 8000² | **22** | 21 | 19 |
| axpy 64M | **11.0** | 4.1 | 11.3 |
| dot 64M | 12.1 | 4.3 | **12.6** |
(NVPL does not thread level-1/2 either; BLIS-gb10 now ~3× NVPL there.)

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
