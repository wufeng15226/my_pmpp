#!/usr/bin/sh

cmake -DCMAKE_CUDA_ARCHITECTURES=all . -B build -G Ninja  &&
cmake --build build &&
# ./build/chapter_02/vectorAdd
# ./build/chapter_03/colorToGrayscaleConversion
# ./build/chapter_03/imageBlur
# ./build/chapter_03/matMul
# ./build/chapter_05/matMulTiled
./build/chapter_06/matMulCornerTurning
./build/chapter_06/matMulThreadCoarse
