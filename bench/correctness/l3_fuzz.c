#include "blis.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double fr(){return (double)rand()/RAND_MAX-0.5;}
static int fails=0;
static void chk(const char*n,double r){if(r>1e-10){printf("  FAIL %s rel=%.2e\n",n,r);fails++;}}
int main(){srand(9);
 dim_t ds[]={1,3,7,13,58,59,127,257,509,1001};int nd=10;
 // syrk: C(lower) = A A^T, m x k
 for(int i=0;i<nd;i++)for(int j=0;j<nd;j++){dim_t m=ds[i],k=ds[j];
   double*A=malloc((size_t)m*k*8),*C=calloc((size_t)m*m,8);for(size_t t=0;t<(size_t)m*k;t++)A[t]=fr();
   double one=1,zero=0; bli_dsyrk(BLIS_LOWER,BLIS_NO_TRANSPOSE,m,k,&one,A,1,m,&zero,C,1,m);
   double e=0,d=0;for(dim_t b=0;b<m;b++)for(dim_t a=b;a<m;a++){double s=0;for(dim_t p=0;p<k;p++)s+=A[a+(size_t)p*m]*A[b+(size_t)p*m];double v=C[a+(size_t)b*m];e+=(v-s)*(v-s);d+=s*s;}
   char nm[48];snprintf(nm,48,"syrk %ldx%ld",(long)m,(long)k);chk(nm,sqrt(e/(d+1e-300)));free(A);free(C);}
 // trsm: solve L X = B (lower, nonunit), m x n; verify L*X == B
 for(int i=0;i<nd;i++)for(int j=0;j<nd;j++){dim_t m=ds[i],n=ds[j];
   double*A=malloc((size_t)m*m*8),*B=malloc((size_t)m*n*8),*B0=malloc((size_t)m*n*8);
   for(size_t t=0;t<(size_t)m*m;t++)A[t]=fr(); for(dim_t a=0;a<m;a++)A[a+(size_t)a*m]+=m+2; // diag dominant
   for(size_t t=0;t<(size_t)m*n;t++){B[t]=fr();B0[t]=B[t];}
   double one=1; bli_dtrsm(BLIS_LEFT,BLIS_LOWER,BLIS_NO_TRANSPOSE,BLIS_NONUNIT_DIAG,m,n,&one,A,1,m,B,1,m);
   // check L*X ?= B0
   double e=0,d=0;for(dim_t c=0;c<n;c++)for(dim_t r=0;r<m;r++){double s=0;for(dim_t p=0;p<=r;p++)s+=A[r+(size_t)p*m]*B[p+(size_t)c*m];double rf=B0[r+(size_t)c*m];e+=(s-rf)*(s-rf);d+=rf*rf;}
   char nm[48];snprintf(nm,48,"trsm %ldx%ld",(long)m,(long)n);chk(nm,sqrt(e/(d+1e-300)));free(A);free(B);free(B0);}
 printf("%s (fails=%d)\n",fails==0?"ALL PASS":"FAILURE",fails);return fails==0?0:1;}
