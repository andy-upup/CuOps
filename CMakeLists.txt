cmake_minimum_required(VERSION 3.30 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(CuOps LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES
    89
    CACHE STRING "" FORCE)

message(STATUS "CMake Version: ${CMAKE_VERSION}")
set(IMPLICIT_CMAKE_CXX_STANDARD
    OFF
    CACHE BOOL "Do not explicitly specify -std=c++17 if set")

# To reduce duplicate version locations, parse the version out of the main
# versions.h file and reuse it here.

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/version.h
     VERSION_FILE_CONTENTS)
string(REGEX MATCH "#define CUTLASS_MAJOR ([0-9]+)" _CUTLASS_VERSION_MAJOR
             "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define CUTLASS_MINOR ([0-9]+)" _CUTLASS_VERSION_MINOR
             "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define CUTLASS_PATCH ([0-9]+)" _CUTLASS_VERSION_PATCH
             "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_PATCH ${CMAKE_MATCH_1})

message(
  STATUS
    "CUTLASS ${_CUTLASS_VERSION_MAJOR}.${_CUTLASS_VERSION_MINOR}.${_CUTLASS_VERSION_PATCH}"
)

find_package(GTest REQUIRED)
find_package(ZLIB REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CUDA.cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(common)
add_subdirectory(examples)
