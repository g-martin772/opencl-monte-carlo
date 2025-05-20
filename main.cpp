#include <iostream>

#include <CL/cl.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "dependencies/yahoo-finance/src/quote.hpp"

#include <TCanvas.h>
#include <TH1F.h>
#include <TRandom3.h>
#include <TGraph.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TROOT.h>

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define N_SIMULATIONS 1000

std::string load_program(std::string input) {
    std::ifstream stream(input.c_str());
    if (!stream.is_open()) {
        std::cout << "Cannot open file: " << input << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(stream),
                       (std::istreambuf_iterator<char>()));
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
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N_SIMULATIONS));
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(float) * N_SIMULATIONS * days, results.data());

    float mean = 0;
    for (auto r: results) mean += r;
    mean /= N_SIMULATIONS * days;

    std::cout << "Expected final price: " << mean << std::endl;
    std::cout << "Total return: " << mean / currentPrice * 100 << "%" << std::endl;

    std::vector<float> means(days, 0), mins(days, std::numeric_limits<float>::max()), maxs(days, 0);
    for (int t = 0; t < days; ++t) {
        for (int i = 0; i < N_SIMULATIONS; ++i) {
            float val = results[i * days + t];
            means[t] += val;
            mins[t] = std::min(mins[t], val);
            maxs[t] = std::max(maxs[t], val);
        }
        means[t] /= N_SIMULATIONS;
    }

    std::vector<float> x_vals(days);
    for (int i = 0; i < days; ++i)
        x_vals[i] = i;

    TApplication app("app", &argc, argv);
    gROOT->SetBatch(kTRUE);

    auto *canvas = new TCanvas("canvas", "Monte Carlo Simulation", 800, 600);
    auto meanGraph = new TGraph(days, x_vals.data(), means.data());
    auto minGraph = new TGraph(days, x_vals.data(), mins.data());
    auto maxGraph = new TGraph(days, x_vals.data(), maxs.data());

    meanGraph->SetLineColor(kBlue);
    minGraph->SetLineColor(kBlack);
    maxGraph->SetLineColor(kRed);

    meanGraph->SetTitle("Mean;Day;Price");
    meanGraph->Draw("AL");
    minGraph->Draw("L SAME");
    maxGraph->Draw("L SAME");

    canvas->BuildLegend();
    canvas->SaveAs("monte_carlo_paths.png");

    delete canvas;
    delete meanGraph;
    delete minGraph;
    delete maxGraph;

    return 0;
}
