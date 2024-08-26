cmake_minimum_required(VERSION 3.26)

project(wei_infer LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_BUILD_TYPE Debug)

# required libs
include(cmake/cuda.cmake)
include(cmake/torch.cmake)

# executable
add_subdirectory(test)

# log
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target architecture: ${CMAKE_CUDA_ARCHITECTURES}")
