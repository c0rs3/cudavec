cmake_minimum_required(VERSION 3.18)
project(cudavec LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/benchtools/src/*.cpp")

add_executable(cudavec
  "${CMAKE_SOURCE_DIR}/main.cpp"
  "${CMAKE_SOURCE_DIR}/cudavec/cudavec.cu"
  ${SRC_FILES}
)

target_include_directories(cudavec PRIVATE
  "${CMAKE_SOURCE_DIR}/benchtools/headers"
  "${CMAKE_SOURCE_DIR}/cudavec"
)

set_target_properties(cudavec PROPERTIES
  CXX_STANDARD 17
  CUDA_STANDARD 14
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES 75
)

message("Compiler options setup done.")

target_link_libraries(cudavec PRIVATE CUDA::cudart CUDA::cublas)
message("CUDA runtime and cublas linked.")
