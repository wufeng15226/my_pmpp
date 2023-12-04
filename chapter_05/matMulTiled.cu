#include<iostream>
#include<cstdlib>
#define N 100
#define M 200
#define K 300

#define TILE_SIZE 32

__global__
void matMulTiledKernel(float* A, float* B, float* C){

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // blockDim.x = blockDim.y = TILE_SIZE
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int taili = threadIdx.y;
    int tailj = threadIdx.x;

    float sum = 0;
    for(int k=0;k<ceil(1.0*M/TILE_SIZE);++k){
        if(i<N && (k*TILE_SIZE+tailj)<M) As[taili][tailj] = A[i*M + k*TILE_SIZE + tailj];
        else As[taili][tailj] = 0;
        if((k*TILE_SIZE+taili)<M && j<K) Bs[taili][tailj] = B[(k*TILE_SIZE + taili)*K + j];
        else Bs[taili][tailj] = 0;
        __syncthreads();
        for(int p=0;p<TILE_SIZE;++p){
            sum += As[taili][p] * Bs[p][tailj];
        }
        __syncthreads();
    }
    if(i<N && j<K) C[i*K + j] = sum;
}

void matMulTiled(float* A, float* B, float* C){
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, N * M * sizeof(float));
    cudaMalloc((void**)&B_d, M * K * sizeof(float));
    cudaMalloc((void**)&C_d, N * K * sizeof(float));
    cudaMemcpy(A_d, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, M * K * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(1.0*K/TILE_SIZE), ceil(1.0*N/TILE_SIZE), 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    matMulTiledKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d);
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
    
    matMulTiled(A, B, C);
    
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