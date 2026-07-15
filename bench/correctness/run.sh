#!/usr/bin/env bash
# Correctness regression for the gb10 tuning. Verifies GEMM and the threaded
# level-1/2/3 ops against naive references across odd/prime dimensions, sizes
# straddling the SUP(256) and small-MT thresholds, transpose combos, alpha/beta,
# strided operands, and multiple thread counts (exercises partition remainders).
set -e
BLIS=${1:-../../lib/gb10/libblis.a}
INC=${2:-../../include/gb10}
CC="gcc -O2 -I $INC"
for t in gemm_fuzz l12_fuzz l3_fuzz; do $CC $t.c -o $t $BLIS -lm -lpthread -fopenmp; done
export OMP_PROC_BIND=close OMP_PLACES=cores
PIN="taskset -c 5-9,15-19"
echo "== GEMM (odd shapes/trans/alpha-beta/strides) =="
$PIN env OMP_NUM_THREADS=1  ./gemm_fuzz
$PIN env OMP_NUM_THREADS=10 ./gemm_fuzz
echo "== level-1/2 threaded (primes/strides/odd shapes) =="
for nt in 10 7 3; do echo -n "nt=$nt: "; $PIN env OMP_NUM_THREADS=$nt ./l12_fuzz; done
echo "== level-3 syrk/trsm (odd shapes) =="
$PIN env OMP_NUM_THREADS=1  ./l3_fuzz
$PIN env OMP_NUM_THREADS=10 ./l3_fuzz
