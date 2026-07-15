# BLIS `gb10` config — GEMM performance on NVIDIA GB10 (DGX Spark)

## Hardware (measured)
- 20 cores: 10× **Cortex-X925** (logical CPUs **5–9, 15–19**, 3.9 GHz) + 10× Cortex-A725 (0–4, 10–14, 2.8 GHz).
- Cortex-X925 (per Chips&Cheese + measurement): **6 FP pipes, all FMA-capable**, 128-bit NEON/SVE2 (SVE VL=128b, no advantage over NEON).
- Caches: L1d 64 KB, **L2 2 MB** (12-cyc), shared L3. L2 read BW ≈ 32 B/cyc.

## Empirical FMA roofline (pure back-to-back `fmla`, memory-free)
| | 1 core | 10× X925 |
|---|---|---|
| FP64 | 89.5 GFLOP/s | 898 GFLOP/s |
| FP32 | 178 GFLOP/s | 1787 GFLOP/s |
(theoretical DP = 6·2·2·3.9 = 93.6 GFLOP/s/core; measured 89.5 = 95.6%. No all-core clock throttling.)

## Results (GFLOP/s, % of FMA roofline). BLIS = this `gb10` config.
### Single core (pinned to X925 core 9)
| | BLIS | NVPL | OpenBLAS |
|---|---|---|---|
| DGEMM 4000 | **84.9 (94.8%)** | 80.2 | 65.6 |
| SGEMM 4000 | **172.5 (96.9%)** | 169.3 | 134.3 |

### 10× X925 (cores 5-9,15-19; BLIS `BLIS_JC_NT=1 BLIS_IC_NT=10`)
| | BLIS | NVPL | OpenBLAS |
|---|---|---|---|
| DGEMM 8000 | **781 (87.0%)** | 721 (80.3%) | 573 |
| SGEMM 8000 | **1631 (91.3%)** | 1588 (88.8%) | 1195 |

**BLIS-gb10 beats NVPL and OpenBLAS at every point measured.** >90% of the FMA roofline
achieved for single-core DP+SP and 10-core SP. 10-core DP tops out ~87% (shared L2/L3
bandwidth bound — even NVPL only reaches 80%).

## Tuning applied
- `config/gb10`: NEON asm kernels `bli_dgemm_armv8a_asm_6x8` / `bli_sgemm_armv8a_asm_8x12`
  (6×8 dgemm tile = 24 vec-reg accumulators = exactly 6 pipes × 4-cyc latency), `-mcpu=gb10`.
- Block sizes (d): MC=336, KC=336, NC=4080 (sized to the 2 MB L2 / shared L3).
- Recommended runtime threading on the P-cluster:
  `taskset -c 5-9,15-19 OMP_NUM_THREADS=10 OMP_PROC_BIND=close OMP_PLACES=cores BLIS_JC_NT=1 BLIS_IC_NT=10`

## Reproduce
Binaries built in this dir: `bench_blis` (native BLIS API), `bench_nvpl`, `bench_openblas`
(CBLAS), `fp_peak`/`fp_peak_s` (roofline). See build commands in git history / driver source.
