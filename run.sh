#!/usr/bin/sh

cmake . -B build -G Ninja &&
cmake --build build &&
./build/chapter_02/vectorAdd