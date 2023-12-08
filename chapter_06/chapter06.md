### 1

``` cpp
__global__
void matMulTiledCornerTuningKernel(float* A, float* B, float* C){

    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int taili = threadIdx.y;
    int tailj = threadIdx.x;

    float sum = 0;
    for(int k=0;k<ceil(1.0*M/32);++k){
        if(i<N && (k*32+tailj)<M) As[taili][tailj] = A[i*M + k*32 + tailj];
        else As[taili][tailj] = 0;
        if(j<K && (k*32+taili)<M) Bs[tailj][taili] = B[j*M + k*32 + taili];
        else Bs[tailj][taili] = 0;
        __syncthreads();
        for(int p=0;p<32;++p){
            sum += As[taili][p] * Bs[tailj][p];
        }
        __syncthreads();
    }
    if(i<N && j<K) C[i*K + j] = sum;
}
```

### 2

I don't get the question.

### 3

#### a

Coalesced.

#### b

Coalescing is not applicable.

#### c

Coalesced.

#### d

Uncoalesced.

#### e

Coalescing is not applicable.

#### f

Coalescing is not applicable.

#### g

Coalesced.

#### h

Coalescing is not applicable.

#### i

Uncoalesced.

### 4

#### a

``` cpp
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
```

$$\frac{2}{2*4B}=0.25OP/B$$

#### b

``` cpp
__global__
void matMulTiledKernel(float* A, float* B, float* C){

    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int taili = threadIdx.y;
    int tailj = threadIdx.x;

    float sum = 0;
    for(int k=0;k<ceil(1.0*M/32);++k){
        if(i<N && (k*32+tailj)<M) As[taili][tailj] = A[i*M + k*32 + tailj];
        else As[taili][tailj] = 0;
        if((k*32+taili)<M && j<K) Bs[taili][tailj] = B[(k*32 + taili)*K + j];
        else Bs[taili][tailj] = 0;
        __syncthreads();
        for(int p=0;p<32;++p){
            sum += As[taili][p] * Bs[p][tailj];
        }
        __syncthreads();
    }
    if(i<N && j<K) C[i*K + j] = sum;
}
```

$$\frac{2*32}{2*4B}=8OP/B$$

#### c

``` cpp
__global__
void matMulThreadCoarseKernel(float* A, float* B, float* C){
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int taili = threadIdx.y;
    int tailj = threadIdx.x;

    float sums[4];
    for(int c=0;c<4;++c){
        sums[c] = 0;
    }
    for(int k=0;k<ceil(1.0*M/32);++k){
        if(i<N && (k*32+tailj)<M) As[taili][tailj] = A[i*M + k*32 + tailj];
        else As[taili][tailj] = 0;
        for(int c=0;c<4;++c){
            int j = tailj + c * 32;
            if((k*32+taili)<M && j<K) Bs[taili][tailj] = B[(k*32 + taili)*K + j];
            else Bs[taili][tailj] = 0;
            __syncthreads();
            for(int p=0;p<32;++p){
                sums[c] += As[taili][p] * Bs[p][tailj];
            }
            __syncthreads();
        }
    }
    for(int c=0;c<4;++c){
        int j = tailj + c * 32;
        if(i<N && j<K) C[i*K + j] = sums[c];
    }
}
```

$$\frac{2*32*4}{(1+4)*4B}=12.8OP/B$$
