#!/usr/bin/env bash
# Shape sweep across BLIS/NVPL/OpenBLAS, pinned to the 10 X925 cores.
export OMP_PROC_BIND=close OMP_PLACES=cores
PIN="taskset -c 5-9,15-19"
NT=10
b(){ $PIN env OMP_NUM_THREADS=$NT BLIS_NUM_THREADS=$NT ./bench_blis "$@"; }
n(){ $PIN env OMP_NUM_THREADS=$NT NVPL_BLAS_NUM_THREADS=$NT ./bench_nvpl "$@"; }
o(){ $PIN env OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT ./bench_openblas "$@"; }
pr=${1:-d}
printf "%-22s %9s %9s %9s  %s\n" "shape (MxNxK)" "BLIS" "NVPL" "OpenBLAS" "win"
run(){ local m=$1 nn=$2 k=$3 lbl=$4
  local B=$(b $pr $m $nn $k); local N=$(n $pr $m $nn $k); local O=$(o $pr $m $nn $k)
  local w=$(awk -v B=$B -v N=$N -v O=$O 'BEGIN{m=B>N?B:N;m=m>O?m:O;print (m==B)?"BLIS":((m==N)?"nvpl":"openblas")}')
  printf "%-22s %9s %9s %9s  %s\n" "${m}x${nn}x${k} $lbl" "$B" "$N" "$O" "$w"
}
echo "### square"; for s in 64 128 256 512 1024 2048 4000; do run $s $s $s; done
echo "### rank-k (MN large, K small)"; for k in 16 32 64 128 256; do run 4000 4000 $k; done
echo "### tall-skinny (M large, N small)"; for nn in 8 32 128; do run 8000 $nn 512; done
echo "### wide (M small, N large)"; for m in 8 32 128; do run $m 8000 512; done
echo "### panel (K large, MN small)"; run 64 64 8000; run 256 256 8000; run 512 512 8000
echo "### small square"; for s in 16 32 48 96 192 384; do run $s $s $s; done
