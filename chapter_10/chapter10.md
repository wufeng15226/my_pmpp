### 1

| iteration | elements | active threads      | divergence warps |
| --------- | -------- | ------------------- | ---------------- |
| 1         | 1024     | 0~511               | 0                |
| 2         | 512      | 0, 2, 4, ... , 510  | 16               |
| 3         | 256      | 0, 4, 8, ... , 508  | 16               |
| 4         | 128      | 0, 8, 16, ... , 504 | 16               |
| 5         | 64       | 0, 16, 32, ..., 496 | 16               |

### 2

| iteration | elements | active threads | divergence warps |
| --------- | -------- | -------------- | ---------------- |
| 1         | 1024     | 0~511          | 0                |
| 2         | 512      | 0~255          | 0                |
| 3         | 256      | 0~127          | 0                |
| 4         | 128      | 0~63           | 0                |
| 5         | 64       | 0~31           | 0                |

### 3

``` c++
__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x;
    for(unsigned in stride = blockDim.x; stride >= 1;stride /= 2) {
        if(i < stride) {
            input[2*blockDim.x-1-i] += input[2*blockDim.x-1-i-stride];
        }
        __syncthreads();
    }
    if(i == 0) {
        *output = input[2*blockDim.x-1];
    }
}
```

### 4

``` cpp
__global__ void CoarsenedMaxReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float mmax = input[i];
    for(unsigned in tile = 1; tile < COARSE_FACTOR*2; ++tile) {
    	mmax = max(mmax, input[i+tile*BLOCK_DIM]);
    }
    input_s[t] = mmax;
    for(unsigned int stride = blockDim.x/2; stride >= 1;stride /= 2) {
        __syncthreads();
        if(t < stride) {
            input_s[t] = max(input_s[t], input_s[t+stride]);
        }
    }
    if(i == 0) {
		atomicMax(output, input_s[0]); // However, atomicMax() seems not avaliable for float.
    }
}
```

### 5

``` cpp
#define N 10000 // not necessarily a multiple of COARSE_FACTOR*2*blockDim.x
__global__ void CoarsenedMaxReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = 0;
    if(i<N){
        sum = input[i];
        for(unsigned in tile = 1; tile < COARSE_FACTOR*2; ++tile) {
            if((i+tile*BLOCK_DIM)<N) sum += input[i+tile*BLOCK_DIM];
        }
    }
    input_s[t] = sum;
    for(unsigned int stride = blockDim.x/2; stride >= 1;stride /= 2) {
        __syncthreads();
        if(t < stride) {
            input_s[t] += input_s[t+stride];
        }
    }
    if(i == 0) {
		atomicAdd(output, input_s[0]);
    }
}
```

### 6

#### a

| iteration | array                             |
| --------- | --------------------------------- |
| 1         | **8** 2 **11** 4 **13** 8 **4** 1 |
| 2         | **19** 2 11 4 **17** 8 4 1        |
| 3         | **36** 2 11 4 17 8 4 1            |

#### b

| iteration | array                  |
| --------- | ---------------------- |
| 1         | **11 10 10 5** 5 8 3 1 |
| 2         | **21 15** 10 5 5 8 3 1 |
| 3         | **36** 15 10 5 5 8 3 1 |
