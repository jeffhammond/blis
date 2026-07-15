// Threaded level-1/2 correctness at prime sizes (stress partition remainders),
// strided operands, and odd gemv/ger shapes with padded leading dims + beta.
#include "blis.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double fr(){return (double)rand()/RAND_MAX-0.5;}
static int fails=0;
static void chk(const char*name,double rel){ if(rel>1e-11){printf("  FAIL %s rel=%.2e\n",name,rel);fails++;} }
int main(){
 srand(777);
 // prime & threshold-straddling lengths, unit and non-unit strides
 dim_t ns[]={1,2,3,7,199999,200003,262139,262147,999983,1000003,4194301};
 int nn=sizeof(ns)/sizeof(ns[0]);
 inc_t incs[]={1,2,3};
 for(int a=0;a<nn;a++)for(int si=0;si<3;si++){
   dim_t n=ns[a]; inc_t s=incs[si];
   double*x=malloc((size_t)n*s*8),*y=malloc((size_t)n*s*8),*y0=malloc((size_t)n*s*8);
   for(size_t i=0;i<(size_t)n*s;i++){x[i]=fr();y[i]=fr();y0[i]=y[i];}
   char nm[64];
   // dot
   double rho,ref=0; for(dim_t i=0;i<n;i++)ref+=x[i*s]*y[i*s];
   bli_ddotv(BLIS_NO_CONJUGATE,BLIS_NO_CONJUGATE,n,x,s,y,s,&rho);
   snprintf(nm,64,"dot n=%ld s=%ld",(long)n,(long)s); chk(nm,fabs(rho-ref)/(fabs(ref)+1e-300));
   // axpy y+=al*x
   double al=1.7; for(dim_t i=0;i<n;i++)y[i*s]=y0[i*s];
   bli_daxpyv(BLIS_NO_CONJUGATE,n,&al,x,s,y,s);
   double e=0,d=0; for(dim_t i=0;i<n;i++){double r=y0[i*s]+al*x[i*s];e+=(y[i*s]-r)*(y[i*s]-r);d+=r*r;}
   snprintf(nm,64,"axpy n=%ld s=%ld",(long)n,(long)s); chk(nm,sqrt(e/(d+1e-300)));
   // nrm2
   double nrm,ss=0; for(dim_t i=0;i<n;i++)ss+=x[i*s]*x[i*s];
   bli_dnormfv(n,x,s,&nrm); snprintf(nm,64,"nrm2 n=%ld s=%ld",(long)n,(long)s); chk(nm,fabs(nrm-sqrt(ss))/(sqrt(ss)+1e-300));
   // asum
   double asm_,aref=0; for(dim_t i=0;i<n;i++)aref+=fabs(x[i*s]);
   bli_dasumv(n,x,s,&asm_); snprintf(nm,64,"asum n=%ld s=%ld",(long)n,(long)s); chk(nm,fabs(asm_-aref)/(aref+1e-300));
   // scal
   for(dim_t i=0;i<n;i++)y[i*s]=x[i*s]; double sc=2.3;
   bli_dscalv(BLIS_NO_CONJUGATE,n,&sc,y,s);
   double se=0,sd=0;for(dim_t i=0;i<n;i++){double r=sc*x[i*s];se+=(y[i*s]-r)*(y[i*s]-r);sd+=r*r;}
   snprintf(nm,64,"scal n=%ld s=%ld",(long)n,(long)s); chk(nm,sqrt(se/(sd+1e-300)));
   free(x);free(y);free(y0);
 }
 // gemv & ger at odd shapes with padded lda + beta, notrans & trans
 dim_t ms[]={1,7,199,1001,4099,8191}, kns[]={1,13,257,999,4093};
 for(int i=0;i<6;i++)for(int j=0;j<5;j++){
   dim_t m=ms[i],n=kns[j]; inc_t lda=m+5;
   double*A=malloc((size_t)lda*n*8);for(size_t t=0;t<(size_t)lda*n;t++)A[t]=fr();
   double al=1.3,be=0.6;
   // notrans: y(m)=be*y+al*A x(n)
   {double*x=malloc(n*8),*y=malloc(m*8),*yr=malloc(m*8);
    for(dim_t t=0;t<n;t++)x[t]=fr(); for(dim_t t=0;t<m;t++){y[t]=fr();yr[t]=y[t];}
    bli_dgemv(BLIS_NO_TRANSPOSE,BLIS_NO_CONJUGATE,m,n,&al,A,1,lda,x,1,&be,y,1);
    double e=0,d=0;for(dim_t r=0;r<m;r++){double s=0;for(dim_t c=0;c<n;c++)s+=A[r+(size_t)c*lda]*x[c];double rf=be*yr[r]+al*s;e+=(y[r]-rf)*(y[r]-rf);d+=rf*rf;}
    char nm[64];snprintf(nm,64,"gemvN m=%ld n=%ld",(long)m,(long)n);chk(nm,sqrt(e/(d+1e-300)));free(x);free(y);free(yr);}
   // trans: y(n)=be*y+al*A^T x(m)
   {double*x=malloc(m*8),*y=malloc(n*8),*yr=malloc(n*8);
    for(dim_t t=0;t<m;t++)x[t]=fr(); for(dim_t t=0;t<n;t++){y[t]=fr();yr[t]=y[t];}
    bli_dgemv(BLIS_TRANSPOSE,BLIS_NO_CONJUGATE,m,n,&al,A,1,lda,x,1,&be,y,1);
    double e=0,d=0;for(dim_t c=0;c<n;c++){double s=0;for(dim_t r=0;r<m;r++)s+=A[r+(size_t)c*lda]*x[r];double rf=be*yr[c]+al*s;e+=(y[c]-rf)*(y[c]-rf);d+=rf*rf;}
    char nm[64];snprintf(nm,64,"gemvT m=%ld n=%ld",(long)m,(long)n);chk(nm,sqrt(e/(d+1e-300)));free(x);free(y);free(yr);}
   // ger col-major: A += al*x y^T
   {double*Ac=malloc((size_t)lda*n*8),*x=malloc(m*8),*yv=malloc(n*8);
    for(size_t t=0;t<(size_t)lda*n;t++)Ac[t]=A[t]; for(dim_t t=0;t<m;t++)x[t]=fr();for(dim_t t=0;t<n;t++)yv[t]=fr();
    bli_dger(BLIS_NO_CONJUGATE,BLIS_NO_CONJUGATE,m,n,&al,x,1,yv,1,Ac,1,lda);
    double e=0,d=0;for(dim_t c=0;c<n;c++)for(dim_t r=0;r<m;r++){double rf=A[r+(size_t)c*lda]+al*x[r]*yv[c];e+=(Ac[r+(size_t)c*lda]-rf)*(Ac[r+(size_t)c*lda]-rf);d+=rf*rf;}
    char nm[64];snprintf(nm,64,"ger m=%ld n=%ld",(long)m,(long)n);chk(nm,sqrt(e/(d+1e-300)));free(Ac);free(x);free(yv);}
   free(A);
 }
 printf("%s (fails=%d)\n", fails==0?"ALL PASS":"FAILURE", fails);
 return fails==0?0:1;
}
