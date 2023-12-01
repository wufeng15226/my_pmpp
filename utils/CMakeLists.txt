add_library(libutils STATIC lodepng.cpp)

target_include_directories(libutils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)