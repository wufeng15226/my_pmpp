#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../utils.h"

__global__
void matMulKernel(half* A, half* B, half* C, int M, int N, int K){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<M && j<N){
        half sum = 0;
        for(int k=0;k<K;++k){
            sum += A[i*K+k] * B[k*N+j];
        }
        C[i*N+j] = sum;
    }
}

void matmul(half* A, half* B, half* C, int M, int N, int K){
    half *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&B_d, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&C_d, M * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(A_d, A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C, K * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 dimGrid(ceil(M/32.0), ceil(N/32.0), 1);
    dim3 dimBlock(32, 32, 1);
    matMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);

    CUDA_CHECK(cudaMemcpy(C, C_d, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}