// Thorough GEMM correctness vs naive reference: odd dims, non-multiples of
// MR/NR/KC, sizes straddling SUP(256) and small-MT thresholds, alpha/beta,
// transpose combos, and column strides (ld > rows).
#include "blis.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double fr(){return (double)rand()/RAND_MAX-0.5;}
// naive C = beta*C + alpha*op(A)*op(B), column-major, ld given
static double check(dim_t m,dim_t n,dim_t k,trans_t ta,trans_t tb,double al,double be){
  // A is (op=notrans? m x k : k x m); allocate with padded leading dims
  dim_t Ar = (ta==BLIS_NO_TRANSPOSE)? m:k, Ac=(ta==BLIS_NO_TRANSPOSE)? k:m;
  dim_t Br = (tb==BLIS_NO_TRANSPOSE)? k:n, Bc=(tb==BLIS_NO_TRANSPOSE)? n:k;
  inc_t lda=Ar+3, ldb=Br+2, ldc=m+1;   // padded leading dims (strided)
  double*A=malloc((size_t)lda*Ac*8),*B=malloc((size_t)ldb*Bc*8),*C=malloc((size_t)ldc*n*8),*R=malloc((size_t)ldc*n*8);
  for(size_t i=0;i<(size_t)lda*Ac;i++)A[i]=fr();
  for(size_t i=0;i<(size_t)ldb*Bc;i++)B[i]=fr();
  for(dim_t j=0;j<n;j++)for(dim_t i=0;i<m;i++){double v=fr();C[i+(size_t)j*ldc]=v;R[i+(size_t)j*ldc]=v;}
  bli_dgemm(ta,tb,m,n,k,&al,A,1,lda,B,1,ldb,&be,C,1,ldc);
  // reference
  #define AE(i,p) ((ta==BLIS_NO_TRANSPOSE)? A[(i)+(size_t)(p)*lda] : A[(p)+(size_t)(i)*lda])
  #define BE(p,j) ((tb==BLIS_NO_TRANSPOSE)? B[(p)+(size_t)(j)*ldb] : B[(j)+(size_t)(p)*ldb])
  double num=0,den=0;
  for(dim_t j=0;j<n;j++)for(dim_t i=0;i<m;i++){
    double s=0; for(dim_t p=0;p<k;p++) s+=AE(i,p)*BE(p,j);
    double ref=be*R[i+(size_t)j*ldc]+al*s;
    double e=C[i+(size_t)j*ldc]-ref; num+=e*e; den+=ref*ref;
  }
  free(A);free(B);free(C);free(R);
  return sqrt(num/(den+1e-300));
}
int main(){
  srand(12345);
  // odd/prime dims, threshold-straddlers, non-multiples of 6/8/12
  dim_t ds[]={1,2,3,5,6,7,8,11,12,13,17,31,37,53,58,59,64,97,127,128,129,199,255,256,257,384,509,700,1001,1500};
  int nd=sizeof(ds)/sizeof(ds[0]);
  trans_t tr[]={BLIS_NO_TRANSPOSE,BLIS_TRANSPOSE};
  double albe[][2]={{1,0},{1,1},{-1,0.5},{2.5,-1.3},{0,1}};
  double worst=0; long tests=0; dim_t wm=0,wn=0,wk=0;
  // full grid over a representative subset of dims x trans x alpha/beta
  dim_t sub[]={1,3,7,13,58,59,127,257,509,1001};
  int ns=sizeof(sub)/sizeof(sub[0]);
  for(int a=0;a<ns;a++)for(int b=0;b<ns;b++)for(int c=0;c<ns;c++)
   for(int ti=0;ti<2;ti++)for(int tj=0;tj<2;tj++){
     int abi=(a+b+c)%5; // vary alpha/beta
     double rel=check(sub[a],sub[b],sub[c],tr[ti],tr[tj],albe[abi][0],albe[abi][1]);
     tests++; if(rel>worst){worst=rel;wm=sub[a];wn=sub[b];wk=sub[c];}
     if(rel>1e-10){printf("  FAIL m=%ld n=%ld k=%ld ti=%d tj=%d al=%g be=%g rel=%.2e\n",
       (long)sub[a],(long)sub[b],(long)sub[c],ti,tj,albe[abi][0],albe[abi][1],rel);}
   }
  // plus a sweep over ALL ds as square-ish odd shapes (m,n,k = d, d+1, d+2)
  for(int i=0;i<nd;i++){dim_t d=ds[i]; double rel=check(d,d+1,d+2,BLIS_NO_TRANSPOSE,BLIS_NO_TRANSPOSE,1.3,0.7);
     tests++; if(rel>worst){worst=rel;wm=d;wn=d+1;wk=d+2;} if(rel>1e-10)printf("  FAIL sq d=%ld rel=%.2e\n",(long)d,rel);}
  printf("%ld tests, worst rel=%.2e at %ldx%ldx%ld -> %s\n",tests,worst,(long)wm,(long)wn,(long)wk,worst<1e-10?"ALL PASS":"FAILURE");
  return worst<1e-10?0:1;
}
