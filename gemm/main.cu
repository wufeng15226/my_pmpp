#include<iostream>
#include<cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "utils.h"

// A M*K
// B K*N
// C M*N
// const int M = 16384;
// const int N = 16384;
// const int K = 16384;
const int M = 1024;
const int N = 1024;
const int K = 1024;

extern void matmul(half *A, half *B, half *C, int M, int N, int K);
extern void cublasMatmul(half *A, half *B, half *C, int M, int N, int K);

int checkError(half *golden, half *C, int M, int N){
    int errors = 0;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float diff = ((float)golden[i * N + j] - (float)C[i * N + j]);
            if (diff < 0)
            {
                diff = -diff;
            }
            float maxv = MAX((float)golden[i * N + j], (float)C[i * N + j]);
            if (maxv < 0)
            {
                maxv = -maxv;
            }
            if (diff / maxv > 1e-2)
            {
                errors += 1;
            }
        }
    }
    return errors;
}

int main(){
    half *A, *B ,*C, *cublasC;
    A = (half*)malloc(M * K * sizeof(half));
    B = (half*)malloc(K * N * sizeof(half));
    C = (half*)malloc(M * N * sizeof(half));
    cublasC = (half*)malloc(M * N * sizeof(half));

    srand(time(NULL));
    #pragma omp parallel for
    for(int i=0; i<M*K; i++){
        A[i] = (half)rand() / (half)(RAND_MAX);
    }
    #pragma omp parallel for
    for(int i=0; i<K*N; i++){
        B[i] = (half)rand() / (half)(RAND_MAX);
    }
    
    cublasMatmul(A, B, cublasC, M, N, K);
    matmul(A, B, C, M, N, K);

    int err = checkError(cublasC, C, M, N);
    if(err) std::cout<<"Error: "<<err<<std::endl;

    free(A);
    free(B);
    free(C);
    free(cublasC);

    return 0;
}