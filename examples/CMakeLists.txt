function(cu_ops_example_add_executable NAME)
  file(GLOB_RECURSE CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

  set(${result_var}
      ${CPP_FILES}
      PARENT_SCOPE)
  message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "NAME: ${NAME}")
  message(STATUS "CU_FILES: ${CU_FILES}")
  add_executable(${NAME} ${CU_FILES})

  target_link_libraries(
    ${NAME} PRIVATE $<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:nvidia::cublas> cuda)
  target_link_libraries(${NAME} PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

  target_compile_features(${NAME} PUBLIC cxx_std_14)

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
  endif()
endfunction()

foreach(EXAMPLE template)
  add_subdirectory(${EXAMPLE})
endforeach()
