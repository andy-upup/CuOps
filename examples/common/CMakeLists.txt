include_directories(${ZLIB_INCLUDE_DIRS})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} COMMON_SRCS)
add_library(common STATIC ${COMMON_SRCS})
target_link_libraries(common PUBLIC ${ZLIB_LIBRARIES})

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
