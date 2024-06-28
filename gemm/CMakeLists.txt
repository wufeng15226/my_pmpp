add_executable(gemmWmma main.cu wmma/gemmWmma.cu cublas/gemmCublas.cu)
target_link_libraries(gemmWmma -lcublas)
# for nvcuda namespace
set_target_properties(gemmWmma PROPERTIES CUDA_ARCHITECTURES "80;90")