#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef CBLAS_HEADER
#define CBLAS_HEADER "cblas.h"
#endif
#include CBLAS_HEADER
static double wt(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static double fr(){return (double)rand()/RAND_MAX-0.5;}
static double* ra(size_t n){double*p=malloc(n*8);for(size_t i=0;i<n;i++)p[i]=fr();return p;}
int main(int c,char**v){
 if(c<3){fprintf(stderr,"usage: %s op m [n] [k] [reps]\n",v[0]);return 1;}
 char*op=v[1]; int m=atoi(v[2]); int n=c>3?atoi(v[3]):m; int k=c>4?atoi(v[4]):m;
 int reps=c>5?atoi(v[5]):0; double best=1e30,flops=0;
 #define TIME(call,fl) do{ call; if(reps<=0){reps=(int)(2e9/(fl)); if(reps<3)reps=3; if(reps>3000)reps=3000;} \
   for(int r=0;r<reps;r++){double t=wt(); call; t=wt()-t; if(t<best)best=t;} flops=(fl);}while(0)
 if(!strcmp(op,"syrk")){ double*A=ra((size_t)m*k),*C=ra((size_t)m*m);
   TIME(cblas_dsyrk(CblasColMajor,CblasLower,CblasNoTrans,m,k,1.0,A,m,0.0,C,m), (double)m*m*k);
 } else if(!strcmp(op,"trsm")){ double*A=ra((size_t)m*m),*B=ra((size_t)m*n);
   for(int i=0;i<m;i++)A[i+i*m]+=m;
   TIME(cblas_dtrsm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,m,n,1.0,A,m,B,m), (double)m*m*n);
 } else if(!strcmp(op,"trmm")){ double*A=ra((size_t)m*m),*B=ra((size_t)m*n);
   TIME(cblas_dtrmm(CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,m,n,1.0,A,m,B,m), (double)m*m*n);
 } else if(!strcmp(op,"gemv")){ double*A=ra((size_t)m*n),*x=ra(n),*y=ra(m);
   TIME(cblas_dgemv(CblasColMajor,CblasNoTrans,m,n,1.0,A,m,x,1,0.0,y,1), 2.0*m*n);
 } else if(!strcmp(op,"axpy")){ double*x=ra(m),*y=ra(m);
   TIME(cblas_daxpy(m,1.0,x,1,y,1), 2.0*m);
 } else if(!strcmp(op,"dot")){ double*x=ra(m),*y=ra(m); volatile double rho;
   TIME(rho=cblas_ddot(m,x,1,y,1), 2.0*m);
 } else { fprintf(stderr,"unknown op\n"); return 1; }
 printf("%.2f", flops/best/1e9); return 0;
}
