// Pure FP32 FMA throughput via inline asm: 30 independent fmla v.4s per iter.
#include <stdio.h>
#include <time.h>
static double wtime(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(){
  const long iters=100000000L;
  double t=wtime();
  __asm__ volatile(
    "fmov v0.4s, #1.0\n fmov v1.4s, #1.0\n"
    "movi v2.4s, #0\n movi v3.4s, #0\n movi v4.4s, #0\n movi v5.4s, #0\n"
    "movi v6.4s, #0\n movi v7.4s, #0\n movi v8.4s, #0\n movi v9.4s, #0\n"
    "movi v10.4s, #0\n movi v11.4s, #0\n movi v12.4s, #0\n movi v13.4s, #0\n"
    "movi v14.4s, #0\n movi v15.4s, #0\n movi v16.4s, #0\n movi v17.4s, #0\n"
    "movi v18.4s, #0\n movi v19.4s, #0\n movi v20.4s, #0\n movi v21.4s, #0\n"
    "movi v22.4s, #0\n movi v23.4s, #0\n movi v24.4s, #0\n movi v25.4s, #0\n"
    "movi v26.4s, #0\n movi v27.4s, #0\n movi v28.4s, #0\n movi v29.4s, #0\n"
    "movi v30.4s, #0\n movi v31.4s, #0\n"
    "mov x0, %0\n"
    "1:\n"
    "fmla v2.4s,v0.4s,v1.4s\n fmla v3.4s,v0.4s,v1.4s\n fmla v4.4s,v0.4s,v1.4s\n fmla v5.4s,v0.4s,v1.4s\n"
    "fmla v6.4s,v0.4s,v1.4s\n fmla v7.4s,v0.4s,v1.4s\n fmla v8.4s,v0.4s,v1.4s\n fmla v9.4s,v0.4s,v1.4s\n"
    "fmla v10.4s,v0.4s,v1.4s\n fmla v11.4s,v0.4s,v1.4s\n fmla v12.4s,v0.4s,v1.4s\n fmla v13.4s,v0.4s,v1.4s\n"
    "fmla v14.4s,v0.4s,v1.4s\n fmla v15.4s,v0.4s,v1.4s\n fmla v16.4s,v0.4s,v1.4s\n fmla v17.4s,v0.4s,v1.4s\n"
    "fmla v18.4s,v0.4s,v1.4s\n fmla v19.4s,v0.4s,v1.4s\n fmla v20.4s,v0.4s,v1.4s\n fmla v21.4s,v0.4s,v1.4s\n"
    "fmla v22.4s,v0.4s,v1.4s\n fmla v23.4s,v0.4s,v1.4s\n fmla v24.4s,v0.4s,v1.4s\n fmla v25.4s,v0.4s,v1.4s\n"
    "fmla v26.4s,v0.4s,v1.4s\n fmla v27.4s,v0.4s,v1.4s\n fmla v28.4s,v0.4s,v1.4s\n fmla v29.4s,v0.4s,v1.4s\n"
    "fmla v30.4s,v0.4s,v1.4s\n fmla v31.4s,v0.4s,v1.4s\n"
    "subs x0,x0,#1\n bne 1b\n"
    : : "r"(iters)
    : "x0","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15",
      "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","cc");
  t=wtime()-t;
  double flop=(double)iters*30*4*2; // 30 fmla * 2 lanes * 2
  printf("FP32 FMA peak: %.2f GFLOP/s (%.2f flop/cyc @3.9GHz => %.1f fmla(4s)/cyc)\n",
         flop/t/1e9, flop/t/3.9e9, flop/t/3.9e9/8.0);
  return 0;
}
