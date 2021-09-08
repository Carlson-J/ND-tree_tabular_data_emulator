//
// Created by jared on 5/26/2021.
//
#define PATH_TO_TEST_DATA "../../Tests/"
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.h"
#include "H5Cpp.h"
#include <highfive/H5File.hpp>
#include "emulator.h"
#include "string"
#include <iostream>
#include <fstream>
#include <omp.h>
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


TEST_CASE("Checkpoint solution", "[checkpoint]"){
    // Load the emulator
//#include "test_v2_sparse_cpp_params.h"

    std::string table_loc = std::string(PATH_TO_TEST_DATA) + std::string("test_table.hdf5");
    std::string mapping_loc = std::string(PATH_TO_TEST_DATA) + std::string("test_mapping.bin");
    auto *emulator = new Emulator<unsigned long, 1, 3, 2555297, 8388608>(table_loc, mapping_loc);
    // create 4d data for interpolation
    const double EPS = 1e-12;
    const size_t NUM_POINTS = 10000;
    const size_t NUM_DIM = 3;
    double temp_min = -3.0;
    double temp_max = 2.45;
    double rho_min = 2.0;
    double rho_max = 16;
    double ye_min = 0.05;
    double ye_max = 0.66;
    double x0[NUM_POINTS];
    double x1[NUM_POINTS];
    double x2[NUM_POINTS];
    double dx[NUM_DIM] = {(ye_max - ye_min)/(NUM_POINTS+10),
                          (temp_max - temp_min)/(NUM_POINTS+10),
                          (rho_max - rho_min)/(NUM_POINTS+10)};
    for (unsigned int i = 0; i < NUM_POINTS; ++i) {
        x0[i] = ye_min + dx[0]*(i+1);
        x1[i] = temp_min + dx[1]*(i+1);
        x2[i] = rho_min + dx[2]*(i+1);
    }
    // Create array of pointers to point to each array
    double* points[NUM_DIM] = {x0, x1, x2};
    // Create array for solution
    double sol[NUM_POINTS] = {0};
    // do interpolation on 4d data
    emulator->interpolate(points, NUM_POINTS, sol);
    // change below to make validation set
    bool make_validation_set = false;
    if (make_validation_set){
        // make validation set
        std::ofstream myfile;
        myfile.open (std::string(PATH_TO_TEST_DATA) + std::string("validation_data.txt"));
        for (size_t i = 0; i < NUM_POINTS; i++){
            myfile << std::scientific << std::setprecision(16) << sol[i] << '\n';
        }
        myfile.close();
    } else{
        // check if results are correct
        std::ifstream myfile;
        myfile.open (std::string(PATH_TO_TEST_DATA) + std::string("validation_data.txt"));
        for (size_t i = 0; i < NUM_POINTS; i++){
    //        double sol_true = x0[i] + x1[i] + x2[i] + 1.0;
    //        REQUIRE(std::fabs(sol[i] - sol_true) < EPS);
            double sol_true = 0;
            std::string word;
            char* pEnd;
            myfile >> word;
            sol_true = std::strtod(word.c_str(), &pEnd);
            REQUIRE(std::fabs(sol[i] - sol_true) < EPS*std::fabs(sol_true));
        }
        myfile.close();
    }

}

TEST_CASE("Checkpoint solution serial", "[checkpoint:serial]"){
    // Load the emulator
//#include "test_v2_sparse_cpp_params.h"

    std::string table_loc = std::string(PATH_TO_TEST_DATA) + std::string("test_table.hdf5");
    std::string mapping_loc = std::string(PATH_TO_TEST_DATA) + std::string("test_mapping.bin");
    auto *emulator = new Emulator<unsigned long, 1, 3, 2555297, 8388608>(table_loc, mapping_loc);
    // create 4d data for interpolation
    const double EPS = 1e-12;
    const size_t NUM_POINTS = 10000;
    const size_t NUM_DIM = 3;
    double temp_min = -3.0;
    double temp_max = 2.45;
    double rho_min = 2.0;
    double rho_max = 16;
    double ye_min = 0.05;
    double ye_max = 0.66;
    double x0[NUM_POINTS];
    double x1[NUM_POINTS];
    double x2[NUM_POINTS];
    double dx[NUM_DIM] = {(ye_max - ye_min)/(NUM_POINTS+10),
                          (temp_max - temp_min)/(NUM_POINTS+10),
                          (rho_max - rho_min)/(NUM_POINTS+10)};
    for (unsigned int i = 0; i < NUM_POINTS; ++i) {
        x0[i] = ye_min + dx[0]*(i+1);
        x1[i] = temp_min + dx[1]*(i+1);
        x2[i] = rho_min + dx[2]*(i+1);
    }
    // Create array of pointers to point to each array
    double* points[NUM_DIM] = {x0, x1, x2};
    // Create array for solution
    double sol[NUM_POINTS] = {0};
    double sol2[NUM_POINTS] = {0};
    double sol2_dy[NUM_POINTS] = {0};
    // do interpolation on 4d data
    for (int i = 0; i < NUM_POINTS; ++i) {
        double point[NUM_DIM];
        for (int j = 0; j < NUM_DIM; j++){
            point[j] = points[j][i];
        }
        emulator->interpolate(point, sol[i]);
        emulator->interpolate<1>(point, sol2[i], sol2_dy[i]);
    }
    // change below to make validation set
    bool make_validation_set = false;
    if (make_validation_set){
        // make validation set
        std::ofstream myfile;
        myfile.open (std::string(PATH_TO_TEST_DATA) + std::string("validation_data.txt"));
        for (size_t i = 0; i < NUM_POINTS; i++){
            myfile << std::scientific << std::setprecision(16) << sol[i] << '\n';
        }
        myfile.close();
    } else{
        // check if results are correct
        std::ifstream myfile;
        myfile.open (std::string(PATH_TO_TEST_DATA) + std::string("validation_data.txt"));
        for (size_t i = 0; i < NUM_POINTS; i++){
            double sol_true = 0;
            std::string word;
            char* pEnd;
            myfile >> word;
            sol_true = std::strtod(word.c_str(), &pEnd);
            REQUIRE(std::fabs(sol[i] - sol_true) < EPS*std::fabs(sol_true));
            REQUIRE(std::fabs(sol2[i] - sol_true) < EPS*std::fabs(sol_true));
        }
        myfile.close();
    }

}


// Need to update all of these ***********************
//TEST_CASE("Load Emulator"){
//    // load the emulator
//    #include "saved_emulator_4d_cpp_params.h"
//    Emulator<ND_TREE_EMULATOR_TYPE> emulator("../../Tests/saved_emulator_4d_table.hdf5");
//    #undef ND_TREE_EMULATOR_TYPE
//}
//
//TEST_CASE("Interpolation on linear function", "[Interp_4d_linear]"){
//    // Load the emulator
//    const double EPS = 1e-12;
//    #include "saved_emulator_4d_cpp_params.h"
//    Emulator<ND_TREE_EMULATOR_TYPE> emulator("../../Tests/saved_emulator_4d_table.hdf5");
//    // create 4d data for interpolation
//    const size_t NUM_POINTS = 4;
//    const size_t NUM_DIM = 4;
//    double x0[NUM_POINTS] = {0.0, 0.3, 0.6, 0.4};
//    double x1[NUM_POINTS] = {0.0, 1.3, 0.1, 2.0};
//    double x2[NUM_POINTS] = {0.0, 3.0, 2.8, 0.1};
//    double x3[NUM_POINTS] = {0.0, 0.0, 4.0, 5.1};
//    // Create array of pointers to point to each array
//    double* points[NUM_DIM] = {x0, x1, x2, x3};
//    // Create array for solution
//    double sol[NUM_POINTS] = {0};
//    // do interpolation on 4d data
//    emulator.interpolate(points, NUM_POINTS, sol);
//    // check if results are correct
//    for (size_t i = 0; i < NUM_POINTS; i++){
//        double sol_true = x0[i] + x1[i] + x2[i] + x3[i] + 1.0;
//        REQUIRE(std::fabs(sol[i] - sol_true) < EPS);
//    }
//}
//
//TEST_CASE("Interpolation on non-linear function", "[Interp_4d_non_linear]"){
//    // Load the emulator
//    const double EPS = 1e-1;
//#include "../../Tests/non_linear2d_cpp_params.h"
//    Emulator<ND_TREE_EMULATOR_TYPE> emulator("../../Tests/non_linear2d_table.hdf5");
//    // create 4d data for interpolation
//    const size_t num_points = 5;
//    const size_t num_dim = 2;
//    double x0[num_points] = {0.0, 0.3, 0.6, 0.4, 1.0};
//    double x1[num_points] = {0.0, 0.56, 0.64, 0.97, 1.0};
//    // Create array of pointers to point to each array
//    double* points[num_dim] = {x0, x1};
//    // Create array for solution
//    double sol[num_points] = {0};
//    // do interpolation on 4d data
//    emulator.interpolate(points, num_points, sol);
//    // check if results are correct
//    for (size_t i = 0; i < num_points; i++){
//        double sol_true = cos(x0[i])*2 + sin(x1[i]);
//        REQUIRE(std::fabs(sol[i] - sol_true) < EPS);
//    }
//}
//
//TEST_CASE("Interpolation on non-linear function in Parallel", "[Interp_2d_non_linear_Parallel]") {
//    // Load the emulator
//    const double EPS = 1e-1;
//
//#include "../../Tests/non_linear2d_cpp_params.h"
//
//    Emulator<ND_TREE_EMULATOR_TYPE> emulator("../../Tests/non_linear2d_table.hdf5");
//    const size_t NUM_RUNS = 100;
//    #pragma omp parallel for default(none) shared(emulator, std::cout)
//    for (size_t j = 0; j < NUM_RUNS; j++){
//        // create 2d data for interpolation
//        const size_t num_points = 5;
//        const size_t num_dim = 2;
//        double x0[num_points] = {0.0, 0.3, 0.6, 0.4, 1.0};
//        double x1[num_points] = {0.0, 0.56, 0.64, 0.97, 1.0};
//        // Create array of pointers to point to each array
//        double *points[num_dim] = {x0, x1};
//        // Create array for solution
//        double sol[num_points] = {0};
//        // do interpolation on 4d data
//        emulator.interpolate(points, num_points, sol);
//        // check if results are correct
//#pragma omp critical
//        for (size_t i = 0; i < num_points; i++) {
//            double sol_true = cos(x0[i])*2 + sin(x1[i]);
//            REQUIRE(std::fabs(sol[i] - sol_true) < EPS);
//        }
//    }
//}
//
//TEST_CASE("Correct mapping in tree index space", "[Index Mapping]"){
//    // Load the emulator
//#include "../../Tests/non_linear2d_cpp_params.h"
//    Emulator<ND_TREE_EMULATOR_TYPE> emulator("../../Tests/non_linear2d_table.hdf5");
//    // create test points
//    const size_t num_points = 5;
//    const size_t num_dim = 2;
//    double x0[num_points] = {0.0, 0.3, 0.6, 0.4, 1.0};
//    double x1[num_points] = {0.0, 0.56, 0.64, 0.97, 1.0};
//    size_t correct_indices[num_points] = {0, 9, 12, 11, 15};
//    double point[num_dim];
//    for (size_t i = 0; i < num_points; i++){
//        point[0] = x0[i];
//        point[1] = x1[i];
//        size_t sol = emulator.compute_tree_index(point);
//        REQUIRE(sol == correct_indices[i]);
//    }
//}