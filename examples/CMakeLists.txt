set(CU_OPS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

function(cu_ops_example_add_executable NAME)
  file(GLOB_RECURSE CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

  add_executable(${NAME} ${CU_FILES})

  target_link_libraries(
    ${NAME} PRIVATE $<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:nvidia::cublas> cuda
                    cu_ops)
  target_link_libraries(${NAME} PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

  target_compile_features(${NAME} PUBLIC cxx_std_14)

  target_include_directories(${NAME}
                             PRIVATE ${CU_OPS_EXAMPLES_COMMON_SOURCE_DIR})

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
  endif()
endfunction()

foreach(EXAMPLE template reduce transpose)
  add_subdirectory(${EXAMPLE})
endforeach()
