#include<iostream>
#include<cstdlib>
#define N 2048
#define BLOCK_SIZE 32
#define TILE_SIZE 64

__global__
void matTransposeSharedMemoryKernel(float* A, float* B){
    unsigned int ti = threadIdx.y;
    unsigned int tj = threadIdx.x;
    __shared__ float A_s[TILE_SIZE][TILE_SIZE];
    // __shared__ float A_s[TILE_SIZE][TILE_SIZE+1]; // no bank confilct
    for(int k=0;k<TILE_SIZE/BLOCK_SIZE;++k){
        for(int p=0;p<TILE_SIZE/BLOCK_SIZE;++p){
            unsigned int i = blockIdx.y*TILE_SIZE+k*BLOCK_SIZE+ti;
            unsigned int j = blockIdx.x*TILE_SIZE+p*BLOCK_SIZE+tj;
            A_s[k*BLOCK_SIZE+ti][p*BLOCK_SIZE+tj] = A[i*N+j];
        }
    }
    __syncthreads();
    for(int k=0;k<TILE_SIZE/BLOCK_SIZE;++k){
        for(int p=0;p<TILE_SIZE/BLOCK_SIZE;++p){
            unsigned int i = blockIdx.x*TILE_SIZE+k*BLOCK_SIZE+ti;
            unsigned int j = blockIdx.y*TILE_SIZE+p*BLOCK_SIZE+tj;
            B[i*N+j] = A_s[p*BLOCK_SIZE+tj][k*BLOCK_SIZE+ti];
        }
    }
}

void matTransposeSharedMemory(float* A, float* B){
    float *A_d, *B_d;
    cudaMalloc((void**)&A_d, N * N * sizeof(float));
    cudaMalloc((void**)&B_d, N * N * sizeof(float));
    cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(1.0*N/TILE_SIZE), ceil(1.0*N/TILE_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    matTransposeSharedMemoryKernel<<<dimGrid, dimBlock>>>(A_d, B_d);
    cudaMemcpy(B, B_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
}

int main(){
    float *A, *B;
    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * N * sizeof(float));

    srand(time(NULL));
    #pragma omp parallel for
    for(int i=0; i<N*N; i++){
        A[i] = (float)rand() / (float)(RAND_MAX);
    }
    
    matTransposeSharedMemory(A, B);
    
    #pragma omp parallel for
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if (abs(A[i*N+j] - B[j*N+i]) > 1e-6){
                std::cout<<i<<' '<<j<<' '<<A[i*N+j]<<' '<<B[j*N+i]<<std::endl;
                break;
            }
        }
        
    }
    
    free(A);
    free(B);

    return 0;
}