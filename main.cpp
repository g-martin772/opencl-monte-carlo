#include <iostream>

#include <CL/cl.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

std::string load_program(std::string input) {
    std::ifstream stream(input.c_str());
    if (!stream.is_open()) {
        std::cout << "Cannot open file: " << input << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(stream),
                       (std::istreambuf_iterator<char>()));
}

int main() {

    cl::Context context(DEVICE);

    // load in kernel source, creating a program object for the context
    cl::Program program(context, load_program("../kernel.cl"), true);

    // get the command queue
    cl::CommandQueue queue(context);

    // create the kernel functor
    auto vadd =
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vadd");


    return 0;
}