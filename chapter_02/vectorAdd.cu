#include<iostream>
#include<cstdlib>
#define N 20000

__global__
void vectorAddKernel(float* A, float* B, float* C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N){
        C[i] = A[i]+B[i];
    }
}

void vectorAdd(float* A, float* B, float* C){
    int size = N * sizeof(float);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    vectorAddKernel<<<ceil(N/256.0), 256>>>(A_d, B_d, C_d);
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(){
    float *A, *B, *C;
    int size = N * sizeof(float);
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    srand(time(NULL));
    #pragma omp parallel for
    for(int i=0; i<N; i++){
        A[i] = ((((float)rand() / (float)(RAND_MAX)) * 100));
		B[i] = ((((float)rand() / (float)(RAND_MAX)) * 100));
    }
    
    vectorAdd(A, B, C);
    
    #pragma omp parallel for
    for(int i=0; i<N; i++){
        if (abs(C[i] - (A[i] + B[i])) > 1e-6){
            std::cout<<"Error: "<<i<<std::endl;
            break;
        }
    }
    
    free(A);
    free(B);
    free(C);

    return 0;
}