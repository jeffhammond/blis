# BLIS `gb10` config — GEMM performance on NVIDIA GB10 (DGX Spark)

## Hardware (measured)
- 20 cores: 10× **Cortex-X925** (logical CPUs **5–9, 15–19**, 3.9 GHz) + 10× Cortex-A725 (0–4, 10–14, 2.8 GHz).
- Cortex-X925 (Chips&Cheese + measured): **6 FP pipes, all FMA-capable**, 128-bit NEON/SVE2 (SVE VL=128b → no advantage over NEON). **A725: 2 FMA pipes.**
- Caches: L1d 64 KB, **L2 2 MB** (12-cyc), shared L3. L2 read BW ≈ 32 B/cyc.
- **Sustainable DRAM bandwidth ≈ 118 GB/s** (STREAM triad), and the 10 X925 cores alone saturate it (all-20 = 117.9 GB/s).

## Empirical FMA roofline (pure back-to-back `fmla`, memory-free)
| | 1 core | 10× X925 | all 20 |
|---|---|---|---|
| FP64 | 89.5 | 898 | 1121 GFLOP/s |
| FP32 | 178 | 1787 | ~2231 GFLOP/s |
(theoretical DP = 6·2·2·3.9 = 93.6 GFLOP/s/core; measured 89.5 = 95.6%. No all-core clock throttling.)

## Results (GFLOP/s, % of FMA roofline). BLIS = this `gb10` config.
### Single core (X925, core 9)
| | BLIS | %peak | NVPL | OpenBLAS |
|---|---|---|---|---|
| DGEMM 4000 | **85.3** | **95.3%** | 80.2 | 65.6 |
| SGEMM 4000 | **172.7** | **97.0%** | 169.3 | 134.3 |

### 10× X925 (cores 5-9,15-19; `BLIS_JC_NT=1 BLIS_IC_NT=10`)
| | BLIS | %peak | NVPL | OpenBLAS |
|---|---|---|---|---|
| DGEMM 8000 | **787** | **87.7%** | 721 (80%) | 573 |
| SGEMM 8000 | **1636** | **91.5%** | 1588 (89%) | 1195 |

**BLIS-gb10 beats NVPL and OpenBLAS at every point measured.** >90% of the FMA
roofline for single-core DP+SP and 10-core SP. 10-core DP tops out ~88%: it is
bounded by shared L3 / DRAM bandwidth within the X925 cluster, not by the FP units
(single-core is 95%) — and NVPL only reaches 80% here.

## Tuning applied
- `config/gb10`: NEON asm kernels `bli_dgemm_armv8a_asm_6x8` / `bli_sgemm_armv8a_asm_8x12`
  (6×8 dgemm tile = 24 vec-reg accumulators = exactly 6 pipes × 4-cyc latency), `-mcpu=gb10`.
- Block sizes (d): MC=336, KC=336, NC=4080 (sized to the 2 MB L2 / shared L3).
- Huge-page memory pool (`BLIS_ENABLE_HUGEPAGE_POOL`): 1 GiB→2 MiB hugetlb, malloc
  fallback. Verified working once the host reserved pages (110×1 GiB + 4096×2 MiB):
  the 17 MiB packed-B pool block lands on 2 MiB pages. **But interleaved A/B
  (`BLI_HP_DISABLE`) shows no measurable DGEMM change — 774.9 (on) vs 775.4 (off).**
  GB10 GEMM is bandwidth-bound, not dTLB-bound: the HW prefetchers hide page-walk
  latency while streaming packed panels, so huge pages neither help nor hurt here.
  (Kept enabled; benefits genuinely TLB-bound workloads and is a no-op otherwise.)
- Recommended runtime threading on the P-cluster:
  `taskset -c 5-9,15-19 OMP_NUM_THREADS=10 OMP_PROC_BIND=close OMP_PLACES=cores BLIS_JC_NT=1 BLIS_IC_NT=10`

## Approaches evaluated and rejected
- **d8×6r microkernel** (more B reuse): worse multicore (754 vs 787) — row-pref C-write cost.
- **All-20 cores (heterogeneous split)**: X925(78.8%) + A725(21.2%) concurrent GEMMs →
  **470 GFLOP/s DP, *worse* than X925-only 787**. Both teams slow ~2× fighting for the
  saturated 118 GB/s DRAM. GB10 GEMM is memory-bound at the full-chip level; the 10
  X925 cores are the fast path.
- **madvise/THP on driver matrices**: no effect (anon THP inactive on this kernel).

## Reproduce
Binaries in this dir: `bench_blis` (native BLIS), `bench_nvpl`, `bench_openblas`
(CBLAS), `fp_peak`/`fp_peak_s` (FMA roofline), `stream` (bandwidth). Pin to X925 with
`taskset -c 5-9,15-19`.
