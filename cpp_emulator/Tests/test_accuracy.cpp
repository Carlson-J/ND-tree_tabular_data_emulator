//
// Created by jared on 5/28/2021.
//

#include "catch.h"
#include "H5Cpp.h"
#include <highfive/H5File.hpp>
#include "emulator.h"
#include <boost/variant.hpp>

TEST_CASE("Testing 2D interpolations", "[Interp 2D Linear]"){
    // Load the emulator
    const double EPS = 1e-12;
    Emulator<int, int> emulator("../../Tests/saved_emulator_4d.hdf5");
    // create 4d data for interpolation
    const size_t num_points = 4;
    const size_t num_dim = 2;
    double x0[num_points] = {0.0, 0.3, 0.6, 0.4};
    double x1[num_points] = {0.0, 1.3, 0.1, 2.0};
    // Create array of pointers to point to each array
    double* points[num_dim] = {x0, x1};
    // Create array for solution
    double sol[num_points] = {0};
    // do interpolation on 4d data
    emulator.interpolate(points, num_points, sol);
    // check if results are correct
    for (size_t i = 0; i < num_points; i++){
        double sol_true = x0[i] + x1[i] + 1.3;
        REQUIRE(std::fabs(sol[i] - sol_true) < EPS);
    }
}