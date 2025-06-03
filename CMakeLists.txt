cmake_minimum_required(VERSION 3.31)
project(opencl_monte_carlo)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCL REQUIRED)

add_executable(opencl_monte_carlo main.cpp)

target_link_libraries(opencl_monte_carlo PRIVATE OpenCL::OpenCL)