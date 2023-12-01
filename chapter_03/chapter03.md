### 1

#### a

``` c++
__global__
void matMulKernel(float* A, float* B, float* C, int width){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<width){
        for(int q=0;q<width;++k){
            float sum = 0;
            for(int k=0;k<width;++k){
                sum += A[i*width+k] * B[k*width+q];
            }
            C[i*width+q] = sum;
        }
    }
}

dim3 dimGrid(1, ceil(width/32.0), 1);
dim3 dimBlock(1, 32, 1);
matMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d);
```

#### b

``` c++
__global__
void matMulKernel(float* A, float* B, float* C, int width){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j<width){
        for(int q=0;q<width;++k){
            float sum = 0;
            for(int k=0;k<width;++k){
                sum += A[q*width+k] * B[k*width+j];
            }
            C[q*width+j] = sum;
        }
    }
}

dim3 dimGrid(ceil(width/32.0), 1, 1);
dim3 dimBlock(32, 1, 1);
matMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d);
```

#### c

...

### 2

``` c++
__global__
void matMulKernel(float* A, float* B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N){
        float sum = 0;
        for(int j=0;j<N;++j){
			sum += B[i*N+j] * C[j];
        }
        A[i] = sum;
    }
}

void matMul(float* A, float* B, float* C, int N){
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, N * sizeof(float));
    cudaMalloc((void**)&B_d, N * N * sizeof(float));
    cudaMalloc((void**)&C_d, N * sizeof(float));
    cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(N/32.0), 1, 1);
    dim3 dimBlock(32, 1, 1);
    matMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, N);
    cudaMemcpy(A, A_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```


### 3

#### a

$$16*32= 512$$

#### b

$$512*(\frac{300-1}{16}+1)*(\frac{150-1}{32}+1)=48640$$

#### c

$$(\frac{300-1}{16}+1)*(\frac{150-1}{32}+1)=95$$

#### d

$$150*300=45000$$

### 4

#### a

$$19*400+9=7609$$

#### b

$$9*500+19=4519$$

### 5

$$5*400*500+20*400+10=1008010$$

### 
