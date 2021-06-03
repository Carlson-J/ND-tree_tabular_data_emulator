#include <iostream>
#include "./emulator_to_profile/non_linear2d_cpp_params.h"
#include "emulator.h"
#include <chrono>

int main() {
    // setup emulator
    auto table_location = "../profiler/emulator_to_profile/non_linear2d_table.hdf5";
    // make emulator
    auto emulator = Emulator<ND_TREE_EMULATOR_TYPE>(table_location);
    // create test data
    // -- scan at a fixed x1
    const size_t num_points = 200;
    double x0[num_points];
    double x1[num_points] = {10};
    double* points[2] = {x0,x1};
    for (size_t i = 0; i < num_points; i++){
        x0[i] = double(i)/num_points;
    }
    double output[num_points] = {0};

    // Start timer
    auto start = std::chrono::steady_clock::now();
    emulator.interpolate(points, num_points, output);
    auto end = std::chrono::steady_clock::now();

    auto diff  = end - start;
    std::cout << "Time: " << std::chrono::duration <double, std::nano> (diff).count() << " ns" << std::endl;
//    for (size_t i = 0; i < num_points; i++){
//        std::cout << output[i] << std::endl;
//    }

}
