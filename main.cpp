#include <iostream>
#include <CL/cl.hpp>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <windows.h>

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define N_SIMULATIONS 10000

std::string load_program(std::string input) {
    std::ifstream stream(input.c_str());
    if (!stream.is_open()) {
        std::cout << "Cannot open file: " << input << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(stream),
                       (std::istreambuf_iterator<char>()));
}

double getTimeInSeconds() {
    static LARGE_INTEGER frequency;
    static BOOL initialized = FALSE;

    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = TRUE;
    }

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);

    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                << " <currentPrice> <expectedReturns> <volatility> <tradingDays> <days>\n";
        return 1;
    }

    float currentPrice = std::atof(argv[1]);
    float expectedReturns = std::atof(argv[2]);
    float volatility = std::atof(argv[3]);
    float tradingDays = std::atof(argv[4]);
    int days = std::atoi(argv[5]);

    std::vector<float> results(N_SIMULATIONS * days);

    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = cl::Device::getDefault();

    cl::Context context(DEVICE);

    cl::Program program(context, load_program("kernel.cl"), true);

    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N_SIMULATIONS * days);

    cl::Kernel kernel(program, "monte_carlo_sim");
    kernel.setArg(0, currentPrice);
    kernel.setArg(1, expectedReturns);
    kernel.setArg(2, volatility);
    kernel.setArg(3, tradingDays);
    kernel.setArg(4, days);
    kernel.setArg(5, output_buffer);

    cl::CommandQueue queue(context, device);
    double start_gpu = getTimeInSeconds();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N_SIMULATIONS));
    queue.finish();
    double end_gpu = getTimeInSeconds();
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float) * N_SIMULATIONS * days, results.data());
    std::cout << "GPU execution time: " << (end_gpu - start_gpu) * 1000 << " ms\n";

    std::ofstream file ("price.csv");
    for (int i = 0; i < N_SIMULATIONS; ++i) {
        for (int j = 0; j < days; ++j) {
            file << results[i * days + j] << ";";
        }
        file << "\n";
    }
    file.close();

    return 0;
}
