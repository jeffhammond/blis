// Pure FP64 FMA throughput via inline asm: 30 independent fmla v.2d per iter.
#include <stdio.h>
#include <time.h>
static double wtime(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(){
  const long iters=100000000L;
  double t=wtime();
  __asm__ volatile(
    "fmov v0.2d, #1.0\n fmov v1.2d, #1.0\n"
    "movi v2.2d, #0\n movi v3.2d, #0\n movi v4.2d, #0\n movi v5.2d, #0\n"
    "movi v6.2d, #0\n movi v7.2d, #0\n movi v8.2d, #0\n movi v9.2d, #0\n"
    "movi v10.2d, #0\n movi v11.2d, #0\n movi v12.2d, #0\n movi v13.2d, #0\n"
    "movi v14.2d, #0\n movi v15.2d, #0\n movi v16.2d, #0\n movi v17.2d, #0\n"
    "movi v18.2d, #0\n movi v19.2d, #0\n movi v20.2d, #0\n movi v21.2d, #0\n"
    "movi v22.2d, #0\n movi v23.2d, #0\n movi v24.2d, #0\n movi v25.2d, #0\n"
    "movi v26.2d, #0\n movi v27.2d, #0\n movi v28.2d, #0\n movi v29.2d, #0\n"
    "movi v30.2d, #0\n movi v31.2d, #0\n"
    "mov x0, %0\n"
    "1:\n"
    "fmla v2.2d,v0.2d,v1.2d\n fmla v3.2d,v0.2d,v1.2d\n fmla v4.2d,v0.2d,v1.2d\n fmla v5.2d,v0.2d,v1.2d\n"
    "fmla v6.2d,v0.2d,v1.2d\n fmla v7.2d,v0.2d,v1.2d\n fmla v8.2d,v0.2d,v1.2d\n fmla v9.2d,v0.2d,v1.2d\n"
    "fmla v10.2d,v0.2d,v1.2d\n fmla v11.2d,v0.2d,v1.2d\n fmla v12.2d,v0.2d,v1.2d\n fmla v13.2d,v0.2d,v1.2d\n"
    "fmla v14.2d,v0.2d,v1.2d\n fmla v15.2d,v0.2d,v1.2d\n fmla v16.2d,v0.2d,v1.2d\n fmla v17.2d,v0.2d,v1.2d\n"
    "fmla v18.2d,v0.2d,v1.2d\n fmla v19.2d,v0.2d,v1.2d\n fmla v20.2d,v0.2d,v1.2d\n fmla v21.2d,v0.2d,v1.2d\n"
    "fmla v22.2d,v0.2d,v1.2d\n fmla v23.2d,v0.2d,v1.2d\n fmla v24.2d,v0.2d,v1.2d\n fmla v25.2d,v0.2d,v1.2d\n"
    "fmla v26.2d,v0.2d,v1.2d\n fmla v27.2d,v0.2d,v1.2d\n fmla v28.2d,v0.2d,v1.2d\n fmla v29.2d,v0.2d,v1.2d\n"
    "fmla v30.2d,v0.2d,v1.2d\n fmla v31.2d,v0.2d,v1.2d\n"
    "subs x0,x0,#1\n bne 1b\n"
    : : "r"(iters)
    : "x0","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15",
      "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","cc");
  t=wtime()-t;
  double flop=(double)iters*30*2*2; // 30 fmla * 2 lanes * 2
  printf("FP64 FMA peak: %.2f GFLOP/s (%.2f flop/cyc @3.9GHz => %.1f fmla/cyc)\n",
         flop/t/1e9, flop/t/3.9e9, flop/t/3.9e9/4.0);
  return 0;
}
