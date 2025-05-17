cmake_minimum_required(VERSION 3.30 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(CuOps LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES
    89
    CACHE STRING "" FORCE)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA.cmake)

add_subdirectory(examples)
