function(cu_ops_example_add_executable NAME)
  file(GLOB_RECURSE CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

  add_executable(${NAME} ${CU_FILES} ${ZLIB_LIBRARIES})

  target_link_libraries(
    ${NAME} PRIVATE $<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:nvidia::cublas> cuda
                    cu_ops common)
  target_link_libraries(${NAME} PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY}
                                        GTest::gtest)

  # target_include_directories(${NAME} PRIVATE
  # ${CU_OPS_EXAMPLES_COMMON_SOURCE_DIR})

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
  endif()
endfunction()

foreach(EXAMPLE template reduce transpose cute)
  add_subdirectory(${EXAMPLE})
endforeach()
