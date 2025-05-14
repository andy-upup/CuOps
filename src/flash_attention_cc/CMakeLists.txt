cmake_minimum_required(VERSION 3.30 FATAL_ERROR)
project(flash_attention)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(flash_attention flash_attention.cu)
set_target_properties(flash_attention PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(flash_attention "${TORCH_LIBRARIES}" -lcudart -lcublas)
set_property(TARGET flash_attention PROPERTY CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(flash_attention PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()