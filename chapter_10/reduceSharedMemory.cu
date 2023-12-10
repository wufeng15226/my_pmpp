#include<iostream>
#include<cstdlib>
#define N 128 // N must be a power of 2

__global__
void reduceSharedMemoryKernel(float* A, float* sum){
    int i = threadIdx.x;
    __shared__ float sA[N/2];
    sA[i] = A[i] + A[i+N/2];
    for(unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        if(i < stride){
            sA[i] += sA[i+stride];
        }
        __syncthreads();
    }
    if(i == 0){
        *sum = sA[0];
    }
}

void reduceSharedMemory(float* A, float* sum){
    float *A_d;
    cudaMalloc((void**)&A_d, (N+1) * sizeof(float));
    cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(N/2, 1, 1);
    reduceSharedMemoryKernel<<<dimGrid, dimBlock>>>(A_d, &A_d[N]);
    cudaMemcpy(sum, &A_d[N], sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d);
}

int main(){
    float *A, sum = 0;
    A = (float*)malloc(N * sizeof(float));

    srand(time(NULL));
    #pragma omp parallel for
    for(int i=0; i<N; i++){
        A[i] = (float)rand() / (float)(RAND_MAX);
    }
    reduceSharedMemory(A, &sum);
    
    float curr = 0;
    #pragma omp parallel for
    for(int i=0; i<N; i++){
        curr += A[i];
    }
    if (abs(curr - sum) > 1e-4){
        std::cout<<"error: "<<curr<<' '<<sum<<std::endl;
    }
    
    free(A);

    return 0;
}