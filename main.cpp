#include <iostream>

#include <CL/cl.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

// #include "dependencies/yahoo-finance/src/quote.hpp"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define N_SIMULATIONS 10000000

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
    // Quote *eurusd = new Quote("AAPL");
    // eurusd->getHistoricalSpots("2018-01-01", "2019-01-10", "1d");
    // eurusd->printSpots();

    auto currentPrice = 211.26f;
    auto expectedReturns = 0.2f;
    auto volatility = 0.3f;
    auto tradingDays = 1.0f / 252.0f;

    std::vector<float> results(N_SIMULATIONS);

    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();

    cl::Context context(DEVICE);

    cl::Program program(context, load_program("../kernel.cl"), true);

    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N_SIMULATIONS);

    cl::Kernel kernel(program, "monte_carlo_sim");
    kernel.setArg(0, currentPrice);
    kernel.setArg(1, expectedReturns);
    kernel.setArg(2, volatility);
    kernel.setArg(3, tradingDays);
    kernel.setArg(4, 2520);
    kernel.setArg(5, output_buffer);

    cl::CommandQueue queue(context, device);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N_SIMULATIONS));
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float) * N_SIMULATIONS, results.data());

    float mean = 0;
    for (auto r: results) mean += r;
    mean /= N_SIMULATIONS;

    std::cout << "Expected final price: " << mean << std::endl;

    return 0;
}
