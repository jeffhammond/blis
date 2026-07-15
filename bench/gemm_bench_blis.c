#include "blis.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double wtime(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double frand(){ return (double)rand()/RAND_MAX-0.5; }
int main(int argc,char**argv){
  if(argc<5){ fprintf(stderr,"usage: %s s|d m n k [reps]\n",argv[0]); return 1; }
  char p=argv[1][0]; dim_t m=atoi(argv[2]),n=atoi(argv[3]),k=atoi(argv[4]);
  int reps=argc>5?atoi(argv[5]):0; double flops=2.0*m*n*k;
  if(reps<=0){ reps=(int)(2e9/flops); if(reps<3)reps=3; if(reps>2000)reps=2000; }
  double best=1e30;
  if(p=='d'){ double one=1.0,zero=0.0;
    double *A=malloc((size_t)m*k*8),*B=malloc((size_t)k*n*8),*C=malloc((size_t)m*n*8);
    for(size_t i=0;i<(size_t)m*k;i++)A[i]=frand(); for(size_t i=0;i<(size_t)k*n;i++)B[i]=frand(); memset(C,0,(size_t)m*n*8);
    bli_dgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,m,n,k,&one,A,1,m,B,1,k,&zero,C,1,m);
    for(int r=0;r<reps;r++){ double t=wtime(); bli_dgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,m,n,k,&one,A,1,m,B,1,k,&zero,C,1,m); t=wtime()-t; if(t<best)best=t; }
  } else { float one=1.0f,zero=0.0f;
    float *A=malloc((size_t)m*k*4),*B=malloc((size_t)k*n*4),*C=malloc((size_t)m*n*4);
    for(size_t i=0;i<(size_t)m*k;i++)A[i]=frand(); for(size_t i=0;i<(size_t)k*n;i++)B[i]=frand(); memset(C,0,(size_t)m*n*4);
    bli_sgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,m,n,k,&one,A,1,m,B,1,k,&zero,C,1,m);
    for(int r=0;r<reps;r++){ double t=wtime(); bli_sgemm(BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,m,n,k,&one,A,1,m,B,1,k,&zero,C,1,m); t=wtime()-t; if(t<best)best=t; }
  }
  printf("%.2f", flops/best/1e9); return 0;
}
