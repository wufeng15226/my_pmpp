#include<iostream>
#include<cstdlib>
#define N 16384 // N must be a power of 2
#define COARSE_FACTOR 4
#define BLOCK_SIZE 1024

__device__
void warpReduce(float* sA, int i){
    unsigned int mask = 0xffffffff;
    float val = sA[i];
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);
    sA[i] = val;
}

__global__
void reduceWarpPrimitivesKernel(float* A, float* sum){
    int i = threadIdx.x;
    int base = blockIdx.x * COARSE_FACTOR * 2 * BLOCK_SIZE;

    float curr = 0;
    for(int j=0;j<2*COARSE_FACTOR;++j){
        curr += A[base+j*BLOCK_SIZE+i];
    }

    __shared__ float sA[BLOCK_SIZE];
    sA[i] = curr;
    #pragma unroll
    for(unsigned int stride = BLOCK_SIZE/2; stride > 16; stride /= 2){
        __syncthreads();
        if(i < stride){
            sA[i] += sA[i+stride];
        }
    }

    if(i<32) warpReduce(sA, i);

    if(i == 0){
        atomicAdd(sum, sA[0]);
    }
}

void reduceWarpPrimitives(float* A, float* sum){
    cudaSetDevice(5);
    float *A_d;
    cudaMalloc((void**)&A_d, (N+1) * sizeof(float));
    cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(N/COARSE_FACTOR/2/BLOCK_SIZE, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    reduceWarpPrimitivesKernel<<<dimGrid, dimBlock>>>(A_d, &A_d[N]);
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
    reduceWarpPrimitives(A, &sum);
    
    float curr = 0;
    #pragma omp parallel for
    for(int i=0; i<N; i++){
        curr += A[i];
    }
    if (abs(curr - sum) > 1e-1){
        std::cout<<"error: "<<curr<<' '<<sum<<std::endl;
    }
    
    free(A);

    return 0;
}