cmake_minimum_required(VERSION 3.27)
project(CUDA-practice LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CUDA_NVCC_FLAGS_DEBUG "-g -G")

add_subdirectory(utils)
add_subdirectory(chapter_02)
add_subdirectory(chapter_03)
add_subdirectory(chapter_05)
add_subdirectory(chapter_06)