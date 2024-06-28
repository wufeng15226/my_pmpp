#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../utils.h"

using namespace nvcuda;

const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

__global__
void matMulKernel(half* A, half* B, half* C, int M, int N, int K){

    int bx = blockIdx.x;
    int by = blockIdx.y;
    // int tx = threadIdx.x;
    // int ty = threadIdx.y;
    // int x = tx * blockDim.x + tx;
    // int y = ty * blockDim.y + ty;

    int wx = bx * wmmaN;
    int wy = by * wmmaM;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, wmmaM, wmmaN, wmmaK, half> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    for(int kk=0;kk<K;kk+=wmmaK){
        wmma::load_matrix_sync(a_frag, A+wy*K+kk, K);
        wmma::load_matrix_sync(b_frag, B+wx*K+kk, K);

        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store the output
    wmma::store_matrix_sync(C+wx*M+wy, c_frag, M, wmma::mem_col_major);
}

void matmul(half* A, half* B, half* C, int M, int N, int K){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    half *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void**)&A_d, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&B_d, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)&C_d, M * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(A_d, A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C, M * N * sizeof(half), cudaMemcpyHostToDevice));

    // C is col major
    dim3 dimBlock(32, 1, 1);
    dim3 dimGrid(ceil(1.0*N/wmmaN), ceil(1.0*M/wmmaM), 1);
    for (int i = 0; i < 2; ++i) matMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);

    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) matMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Running cost (ms) of Wmma is " << ms / 10.0 << "\n";
    std::cout << "TFLOPS: " << (float)M * N * K * 2 / (ms / 10.0) * 1e3 / 1e12 << "\n";

    CUDA_CHECK(cudaMemcpy(C, C_d, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}