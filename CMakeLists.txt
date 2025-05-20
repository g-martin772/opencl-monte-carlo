cmake_minimum_required(VERSION 3.31)
project(opencl_monte_carlo)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCL REQUIRED)
find_package(CURL REQUIRED)
find_package(ROOT REQUIRED COMPONENTS Hist Gpad Graf RIO Core)

add_subdirectory(dependencies/yahoo-finance)

add_executable(opencl_monte_carlo main.cpp)

target_link_libraries(opencl_monte_carlo PRIVATE OpenCL::OpenCL)
target_link_libraries(opencl_monte_carlo PRIVATE yahoo-finance CURL::libcurl)
target_link_libraries(opencl_monte_carlo PRIVATE ROOT::Hist ROOT::Gpad ROOT::Graf ROOT::RIO ROOT::Core)