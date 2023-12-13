cmake_minimum_required(VERSION 3.8)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc)
project(ROM LANGUAGES CXX CUDA)


#Set Cmake build type to by defualt to release
if(NOT CMAKE_BUILD_TYPE)
 set(CMAKE_BUILD_TYPE Release)
endif()

#Set the cmake flags as O3 always
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_language(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17" )
add_executable(rom launch_managed.cu rom.cu)