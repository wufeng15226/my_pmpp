cmake_minimum_required(VERSION 3.27)
project(CUDA-practice LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")

add_subdirectory(utils)
add_subdirectory(chapter_02)
add_subdirectory(chapter_03)
add_subdirectory(chapter_05)
add_subdirectory(chapter_06)
add_subdirectory(chapter_10)
add_subdirectory(gemm)