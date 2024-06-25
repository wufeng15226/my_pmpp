#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "../utils.h"

inline cublasStatus_t
gemm(cublasHandle_t handle,
     cublasOperation_t transA, cublasOperation_t transB,
     int m, int n, int k,
     const float* alpha,
     const half* A, int ldA,
     const half* B, int ldB,
     const float* beta,
     half* C, int ldC)
{
  return cublasGemmEx(handle, transA, transB,
                      m, n, k,
                      reinterpret_cast<const float*>(alpha),
                      reinterpret_cast<const __half*>(A), CUDA_R_16F, ldA,
                      reinterpret_cast<const __half*>(B), CUDA_R_16F, ldB,
                      reinterpret_cast<const float*>(beta),
                      reinterpret_cast<      __half*>(C), CUDA_R_16F, ldC,
                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}


void cublasMatmul(half* hA, half* hB, half* hC, int M, int N, int K){

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float alpha = 1.0;
    float beta = 0.0;

    half *dA;
    half *dB;
    half *dC;

    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, K * N * 2));
    CUDA_CHECK(cudaMalloc(&dC, M * N * 2));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * 2, cudaMemcpyHostToDevice));

    // warmup
    for (int i = 0; i < 10; ++i)
    {
        gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    }

    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < 200; ++i)
    {
        gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Running cost (ms) of CuBLAS is " << ms / 200.0 << "\n";
    std::cout << "TFLOPS: " << (float)M * N * K * 2 / (ms / 200.0) * 1e3 / 1e12 << "\n";

    cudaMemcpy(hC, dC, M * K * 2, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}