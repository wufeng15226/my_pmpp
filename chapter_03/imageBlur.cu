#include <iostream>
#include "lodepng/lodepng.h"

#define CHANNELS 4
#define BLUR_SIZE 2

#define IN_FILE "chapter_03/png/violet.png"
#define OUT_FILE "chapter_03/png/violet_blur.png"

__global__
void imageBlurKernel(unsigned char* in, unsigned char* out, unsigned int m, unsigned int n){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n && j<m){
        int offset = CHANNELS * (j + i * m);
        int pixels = 0;
        int red = 0;
        int green = 0;
        int blue = 0;
        for(int add_i = -BLUR_SIZE; add_i <= BLUR_SIZE; add_i++){
            for(int add_j = -BLUR_SIZE; add_j <= BLUR_SIZE; add_j++){
                int current_i = i + add_i;
                int current_j = j + add_j;
                if(current_i >= 0 && current_i < n && current_j >= 0 && current_j < m){
                    int offset = CHANNELS * (current_j + current_i * m);
                    red += in[offset];
                    green += in[offset+1];
                    blue += in[offset+2];
                    pixels++;
                }
            }
        }
        
        out[offset] = red / pixels;
        out[offset+1] = green / pixels;
        out[offset+2] = blue / pixels;
        out[offset+3] = 255;
    }
}

void imageBlur(unsigned char* in, unsigned char* out, unsigned int m, unsigned int n){
    int size = sizeof(unsigned char) * m * n * CHANNELS;
    unsigned char *in_d, *out_d;
    cudaMalloc((void**)&in_d, size);
    cudaMalloc((void**)&out_d, size);
    cudaMemcpy(in_d, in, size, cudaMemcpyHostToDevice);
    
    dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    imageBlurKernel<<<dimGrid, dimBlock>>>(in_d, out_d, m, n);
    
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
    
    imageBlur(in_h, out_h, m, n);
    lodepng_encode32_file(OUT_FILE, out_h, m, n);
        
    free(in_h);
    free(out_h);
    
    return 0;
}