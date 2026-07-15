// Simple STREAM triad a[i]=b[i]+s*c[i], OpenMP, to gauge sustainable BW.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
static double wt(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int c,char**v){ long N=v[1]?atol(v[1]):200000000L; double s=3.0;
  double *a=malloc(N*8),*b=malloc(N*8),*cc=malloc(N*8);
  #pragma omp parallel for
  for(long i=0;i<N;i++){a[i]=0;b[i]=1;cc[i]=2;}
  double best=1e30;
  for(int r=0;r<10;r++){ double t=wt();
    #pragma omp parallel for
    for(long i=0;i<N;i++) a[i]=b[i]+s*cc[i];
    t=wt()-t; if(t<best)best=t; }
  double gb=3.0*N*8/best/1e9; printf("STREAM triad: %.1f GB/s\n",gb); return a[0]==-1?1:0;
}
