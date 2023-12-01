#include<iostream>
#include<cstdlib>
#define N 100
#define M 200
#define K 300

__global__
void matMulKernel(float* A, float* B, float* C){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N && j<K){
        float sum = 0;
        for(int k=0;k<M;++k){
            sum += A[i*M+k] * B[k*K+j];
        }
        C[i*K+j] = sum;
    }
}

void matMul(float* A, float* B, float* C){
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, N * M * sizeof(float));
    cudaMalloc((void**)&B_d, M * K * sizeof(float));
    cudaMalloc((void**)&C_d, N * K * sizeof(float));
    cudaMemcpy(A_d, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, M * K * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(K/32.0), ceil(N/32.0), 1);
    dim3 dimBlock(32, 32, 1);
    matMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d);
    cudaMemcpy(C, C_d, N * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    float *A, *B ,*C;
    A = (float*)malloc(N * M * sizeof(float));
    B = (float*)malloc(M * K * sizeof(float));
    C = (float*)malloc(N * K * sizeof(float));

    srand(time(NULL));
    #pragma omp parallel for
    for(int i=0; i<N*M; i++){
        A[i] = (float)rand() / (float)(RAND_MAX);
    }
    #pragma omp parallel for
    for(int i=0; i<M*K; i++){
        B[i] = (float)rand() / (float)(RAND_MAX);
    }
    
    matMul(A, B, C);
    
    #pragma omp parallel for
    for(int i=0; i<N; i++){
        for(int j=0; j<K; j++){
            float sum = 0;
            for(int k=0;k<M;++k){
                sum += A[i*M+k] * B[k*K+j];
            }
            // in fact, no need to compare, see artical below, especially 5.4 
            // https://docs.nvidia.com/cuda/floating-point/#verifying-gpu-results
            if (abs(C[i*K+j] - sum) > 1e-4){
                std::cout<<i<<' '<<j<<' '<<C[i*K+j]<<' '<<sum<<' '<<abs(C[i*K+j] - sum)<<std::endl;
                break;
            }
        }
        
    }
    
    free(A);
    free(B);
    free(C);

    return 0;
}