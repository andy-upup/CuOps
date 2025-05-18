cmake_minimum_required(VERSION 3.30 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(CuOps LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES
    89
    CACHE STRING "" FORCE)

find_package(GTest REQUIRED)
find_package(ZLIB REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA.cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(examples)
