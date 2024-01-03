#include<iostream>
#include<cstdlib>
#define N 4096 // N must be a power of 2
#define COARSE_FACTOR 4
#define BLOCK_SIZE 128

__global__
void reduceThreadCoarseKernel(float* A, float* sum){
    int i = threadIdx.x;
    int base = blockIdx.x * COARSE_FACTOR * 2 * BLOCK_SIZE;

    float curr = 0;
    for(int j=0;j<2*COARSE_FACTOR;++j){
        curr += A[base+j*BLOCK_SIZE+i];
    }

    __shared__ float sA[BLOCK_SIZE];
    sA[i] = curr;
    for(unsigned int stride = BLOCK_SIZE/2; stride >= 1; stride /= 2){
        __syncthreads();
        if(i < stride){
            sA[i] += sA[i+stride];
        }
    }
    if(i == 0){
        atomicAdd(sum, sA[0]);
    }
}

void reduceThreadCoarse(float* A, float* sum){
    float *A_d;
    cudaMalloc((void**)&A_d, (N+1) * sizeof(float));
    cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(N/COARSE_FACTOR/2/BLOCK_SIZE, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    reduceThreadCoarseKernel<<<dimGrid, dimBlock>>>(A_d, &A_d[N]);
    cudaMemcpy(sum, &A_d[N], sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d);
}

int main(){
    float *A, sum = 0;
    A = (float*)malloc( (N+1) * sizeof(float));

    srand(time(NULL));
    #pragma omp parallel for
    for(int i=0; i<N; i++){
        A[i] = (float)rand() / (float)(RAND_MAX);
    }
    A[N] = 0;
    reduceThreadCoarse(A, &sum);
    
    float curr = 0;
    #pragma omp parallel for
    for(int i=0; i<N; i++){
        curr += A[i];
    }
    if (abs(curr - sum) > 1e-2){
        std::cout<<"error: "<<curr<<' '<<sum<<std::endl;
    }
    
    free(A);

    return 0;
}