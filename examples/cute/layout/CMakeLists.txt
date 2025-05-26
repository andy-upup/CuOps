# Get the full path of the directory where the current file is located.
get_filename_component(CURRENT_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)

# Extract the directory name from the full path.
get_filename_component(CURRENT_DIR_NAME ${CURRENT_DIR} NAME)

cu_ops_example_add_executable(${CURRENT_DIR_NAME} ${CURRENT_DIR_NAME}.cu)
