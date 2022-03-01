cmake_minimum_required(VERSION 3.16)
project(CUDA_DEMO VERSION 1.0.0 LANGUAGES CUDA)


add_subdirectory(my-libs)
add_subdirectory(lib-exe)

#find_package(CUDA REQUIRED)
#include_directories(${CUDA_INCLUDE_DIRS})
#
#link_directories(${CUDA_LIBRARIES})


#set(CMAKE_CUDA_STANDARD 11)

#add_executable(cudaexe lib-exe/main.cu my-libs/src/my-libs/functions.cu my-libs/src/my-libs/functions.cuh)
#
#set_target_properties(
#    cudaexe
#    PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON)