//
// Created by jared on 5/26/2021.
//

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.h"
#include "H5Cpp.h"
#include <highfive/H5File.hpp>
#include "emulator.h"

unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Test the catch testing framework", "[test_framework]" ) {
REQUIRE( Factorial(1) == 1 );
REQUIRE( Factorial(2) == 2 );
REQUIRE( Factorial(3) == 6 );
REQUIRE( Factorial(10) == 3628800 );
}

TEST_CASE("Test hdf5 read and write", "[HDF5]"){
    using namespace HighFive;
    // we create a new hdf5 file
    File file("new_file.h5", File::ReadWrite | File::Create | File::Truncate);

    std::vector<int> data(50, 1);
    std::vector<int> data2(50, 2);

    // let's create a dataset of native integer with the size of the vector 'data'
    DataSet dataset = file.createDataSet<int>("./dataset_one.hdf5",  DataSpace::From(data));

    // let's write our vector of int to the HDF5 dataset
    dataset.write(data);

    // read back
    std::vector<int> result;
    dataset.read(result);


    // Test if it is the same
    REQUIRE(data == result);
    REQUIRE_FALSE(data2 == result);
}

TEST_CASE("Load Emulator"){
    // load the emulator
    Emulator<int, int> emulator("../../Tests/saved_emulator.hdf5");
}

TEST_CASE("Interpolation return size and type", "[Interp1]"){
    // Load the emulator
    Emulator<int, int> emulator("../../Tests/saved_emulator.hdf5");
    // create 4d data for interpolation
    const size_t num_points = 3;
    const size_t num_dim = 4;
    double x0[num_points] = {1.0, 2.0, 4.0};
    double x1[num_points] = {10.0, 20.0, 40.0};
    double x2[num_points] = {-1.0, -2.0, -4.0};
    double x3[num_points] = {-10.0, -20.0, -40.0};
    // Create array of pointers to point to each array
    double* points[num_dim] = {x0, x1, x2, x3};
    // Create array for solution
    double sol[num_points] = {0};
    // do interpolation on 4d data
    emulator.interpolate(points, num_points, sol);
}