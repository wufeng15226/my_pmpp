add_executable(gemmWmma main.cu wmma/gemmWmma.cu cublas/gemmCublas.cu)
target_link_libraries(gemmWmma -lcublas)