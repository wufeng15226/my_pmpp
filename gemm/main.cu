#include<iostream>
#include<cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "utils.h"

// #define DEBUG
// #define PRINT

// A M*K row major
// B K*N col major
// C M*N col major
// const int M = 16384;
// const int N = 16384;
// const int K = 16384;
const int M = 1024;
const int N = 1024;
const int K = 1024;
// const int M = 128;
// const int N = 128;
// const int K = 128;

extern void matmul(half *A, half *B, half *C, int M, int N, int K);
extern void cublasMatmul(half *A, half *B, half *C, int M, int N, int K);
void CPUMatmul(half *A, half *B, half *golden, int M, int N, int K){
#pragma omp parallel for
    for (int i = 0; i < M; i += 64)
    {
#pragma omp parallel for
        for (int j = 0; j < N; j += 64)
        {
            float accum[64 * 64] = {0};
            for (int k = 0; k < K; k += 32)
            {
                for (int kk = 0; kk < 32; ++kk)
                {
                    for (int jj = 0; jj < 64; ++jj)
                    {
                        for (int ii = 0; ii < 64; ++ii)
                        {
                            accum[ii * 64 + jj] += ((float)A[(i + ii) * K + k + kk] * (float)B[(j + jj) * K + k + kk]);
                        }
                    }
                }
            }
            for (int ii = 0; ii < 64; ++ii)
            {
                for (int jj = 0; jj < 64; ++jj)
                {
                    // golden is row major
                    golden[(i + ii) * N + j + jj] = (half)accum[jj * 64 + ii];
                }
            }
        }
    }
    std::cout << "Golden values done!\n";
}

int checkError(half *golden, half *C, int M, int N){
    int errors = 0;
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            float diff = ((float)golden[j * M + i] - (float)C[j * M + i]);
            if (diff < 0)
            {
                diff = -diff;
            }
            float maxv = MAX((float)golden[j * M + i], (float)C[j * M + i]);
            if (maxv < 0)
            {
                maxv = -maxv;
            }
            if (diff / maxv > 1e-2)
            {
                errors += 1;
                std::cout<<i<<' '<<j<<' '<<(float)golden[j * M + i]<<' '<<(float)C[j * M + i]<<std::endl;
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
        A[i] = (half)((rand() % 20+ 0.0));
    }
    #pragma omp parallel for
    for(int i=0; i<K*N; i++){
        B[i] = (half)((rand() % 20+ 0.0));
    }
    #pragma omp parallel for
    for(int i=0; i<M*N; i++){
        C[i] = (half)(0.0);
    }
    #pragma omp parallel for
    for(int i=0; i<M*N; i++){
        cublasC[i] = (half)(0.0);
    }

#ifdef DEBUG
    half* goldenC = (half*)malloc(M * N * sizeof(half));
    #pragma omp parallel for
    for(int i=0; i<M*N; i++){
        goldenC[i] = (half)(0.0);
    }
#else
#endif

#ifdef PRINT
    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            std::cout << (float)(A[i*K+j]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for(int j=0;j<N;j++){
        for(int i=0;i<K;i++){
            std::cout << (float)(B[j*K+i]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
#else
#endif

    // A row major, B col major
    cublasMatmul(A, B, cublasC, M, N, K);
    matmul(A, B, C, M, N, K);

#ifdef DEBUG
    CPUMatmul(A, B, goldenC, M, N, K);
#else
#endif

#ifdef PRINT
    for(int j=0;j<N;j++){
        for(int i=0;i<M;i++){
            std::cout << (float)(C[j*M+i]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for(int j=0;j<N;j++){
        for(int i=0;i<M;i++){
            std::cout << (float)(cublasC[j*M+i]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
#ifdef DEBUG
    for(int j=0;j<N;j++){
        for(int i=0;i<M;i++){
            std::cout << (float)(goldenC[j*M+i]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
#else
#endif
#else
#endif

    int err = checkError(cublasC, C, M, N);
#ifdef DEBUG
    err += checkError(goldenC, C, M, N);
#else
#endif
    if(err) std::cout<<"Error: "<<err<<std::endl;
    else std::cout<<"Pass!"<<std::endl;

    free(A);
    free(B);
    free(C);
    free(cublasC);
#ifdef DEBUG
    free(goldenC);
#else
#endif

    return 0;
}