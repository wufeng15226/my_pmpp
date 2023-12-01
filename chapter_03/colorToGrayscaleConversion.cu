#include <iostream>
#include "lodepng/lodepng.h"

#define CHANNELS 4

#define IN_FILE "chapter_03/png/violet.png"
#define OUT_FILE "chapter_03/png/violet_gray.png"

__global__
void colorToGrayscaleConversionKernel(unsigned char* in, unsigned char* out, unsigned int m, unsigned int n){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<n && j<m){
        int offset = CHANNELS * (j + i * m);
        out[offset] = 0.21 * in[offset] + 0.71 * in[offset+1] + 0.07 * in[offset+2];
        
        out[offset+1] = out[offset];
        out[offset+2] = out[offset];
        out[offset+3] = 255;
    }
}

void colorToGrayscaleConversion(unsigned char* in, unsigned char* out, unsigned int m, unsigned int n){
    int size = sizeof(unsigned char) * m * n * CHANNELS;
    unsigned char *in_d, *out_d;
    cudaMalloc((void**)&in_d, size);
    cudaMalloc((void**)&out_d, size);
    cudaMemcpy(in_d, in, size, cudaMemcpyHostToDevice);
    
    dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    colorToGrayscaleConversionKernel<<<dimGrid, dimBlock>>>(in_d, out_d, m, n);
    
    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
}

int main(){
    const char *filename = IN_FILE;

    unsigned char *in_h, *out_h;
    unsigned int m, n;
    lodepng_decode32_file(&in_h, &m, &n, filename);
    int size = sizeof(unsigned char) * m * n * CHANNELS;
    out_h = (unsigned char*)malloc(size);
    
    colorToGrayscaleConversion(in_h, out_h, m, n);
    lodepng_encode32_file(OUT_FILE, out_h, m, n);
        
    free(in_h);
    free(out_h);
    
    return 0;
}