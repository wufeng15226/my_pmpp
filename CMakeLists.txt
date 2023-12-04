cmake_minimum_required(VERSION 3.23)
project(CUDA-practice LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED 17)
cuda_select_nvcc_arch_flags(ARCH_FLAGS 8.0+PTX 8.0) # -arch=sm_80 / -arch=sm_80 -code=compute_80,sm_80
# set(CUDA_NVCC_FLAGS_DEBUG "-g -G")

add_subdirectory(utils)
add_subdirectory(chapter_02)
add_subdirectory(chapter_03)
add_subdirectory(chapter_05)