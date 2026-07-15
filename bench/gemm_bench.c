// Generic CBLAS GEMM benchmark. Link against BLIS, NVPL, or OpenBLAS.
// Usage: ./bench <s|d> <m> <n> <k> [reps]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef CBLAS_HEADER
#define CBLAS_HEADER "cblas.h"
#endif
#include CBLAS_HEADER
static double wtime(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double frand(){ return (double)rand()/RAND_MAX-0.5; }
int main(int argc,char**argv){
  if(argc<5){ fprintf(stderr,"usage: %s s|d m n k [reps]\n",argv[0]); return 1; }
  char p=argv[1][0]; int m=atoi(argv[2]),n=atoi(argv[3]),k=atoi(argv[4]);
  int reps=argc>5?atoi(argv[5]):0;
  double flops=2.0*m*n*k;
  if(reps<=0){ double target=2e9; reps=(int)(target/flops); if(reps<3)reps=3; if(reps>2000)reps=2000; }
  double best=1e30;
  if(p=='d'){
    double *A=malloc((size_t)m*k*8),*B=malloc((size_t)k*n*8),*C=malloc((size_t)m*n*8);
    for(size_t i=0;i<(size_t)m*k;i++)A[i]=frand(); for(size_t i=0;i<(size_t)k*n;i++)B[i]=frand();
    memset(C,0,(size_t)m*n*8);
    // warmup
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,A,m,B,k,0.0,C,m);
    for(int r=0;r<reps;r++){ double t=wtime();
      cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0,A,m,B,k,0.0,C,m);
      t=wtime()-t; if(t<best)best=t; }
  } else {
    float *A=malloc((size_t)m*k*4),*B=malloc((size_t)k*n*4),*C=malloc((size_t)m*n*4);
    for(size_t i=0;i<(size_t)m*k;i++)A[i]=frand(); for(size_t i=0;i<(size_t)k*n;i++)B[i]=frand();
    memset(C,0,(size_t)m*n*4);
    cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,A,m,B,k,0.0f,C,m);
    for(int r=0;r<reps;r++){ double t=wtime();
      cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1.0f,A,m,B,k,0.0f,C,m);
      t=wtime()-t; if(t<best)best=t; }
  }
  printf("%.2f", flops/best/1e9); // GFLOP/s
  return 0;
}
