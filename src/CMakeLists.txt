set(LIB_NAME cu_ops)

file(GLOB_RECURSE CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

add_library(${LIB_NAME} SHARED ${CU_FILES})
set_target_properties(${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(${LIB_NAME} PUBLIC cxx_std_14)
target_link_libraries(${LIB_NAME} PUBLIC -lcudart -lcublas)
