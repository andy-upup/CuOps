if(CUDA_COMPILER MATCHES "[Cc]lang")
  message(
    WARNING
      "CUDA_COMPILER flag is deprecated, set CMAKE_CUDA_COMPILER to desired compiler executable."
  )
  set(__CLANG_DEVICE_COMPILATION_REQUESTED ON)
elseif(CUDA_COMPILER)
  message(
    WARNING
      "Deprecated flag CUDA_COMPILER used with unknown argument ${CUDA_COMPILER}, ignoring."
  )
endif()

if(__CLANG_DEVICE_COMPILATION_REQUESTED AND NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER clang++) # We will let the system find Clang or error
                                   # out
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

if(NOT CUDA_VERSION)
  # For backward compatibility with older CMake code.
  set(CUDA_VERSION ${CUDAToolkit_VERSION})
  set(CUDA_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR})
  set(CUDA_VERSION_MINOR ${CUDAToolkit_VERSION_MINOR})
endif()
if(NOT CUDA_TOOLKIT_ROOT_DIR)
  # In some scenarios, such as clang device compilation, the toolkit root may
  # not be set, so we force it here to the nvcc we found via the CUDAToolkit
  # package.
  get_filename_component(CUDA_TOOLKIT_ROOT_DIR
                         "${CUDAToolkit_NVCC_EXECUTABLE}/../.." ABSOLUTE)
endif()

if(CUDA_VERSION VERSION_LESS 9.2)
  message(FATAL_ERROR "CUDA 9.2+ required, found ${CUDA_VERSION}.")
endif()

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
