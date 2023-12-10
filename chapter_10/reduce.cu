#include<iostream>
#include<cstdlib>
#define N 128 // N must be a power of 2

__global__
void reduceKernel(float* A, float* sum){
    int i = threadIdx.x;
    for(unsigned int stride = blockDim.x; stride >= 1; stride /= 2){
        if(i < stride){
            A[i] += A[i+stride];
        }
        __syncthreads();
    }
    if(i == 0){
        *sum = A[0];
    }
}

void reduce(float* A, float* sum){
    float *A_d;
    cudaMalloc((void**)&A_d, (N+1) * sizeof(float));
    cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(N/2, 1, 1);
    reduceKernel<<<dimGrid, dimBlock>>>(A_d, &A_d[N]);
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
    reduce(A, &sum);
    
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