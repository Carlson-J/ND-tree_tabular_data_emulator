//
// Created by jared on 5/31/2021.
//
#include "emulator.h"
#include "table_params.h"

// #define POINT_INPUTS double* x, double* y, double* z
// #define POINT_INPUTS_SINGLE double& x, double& y, double& z
// #define POINT_GROUPING double* points[3] = {x,y,z}
// #define POINT_GROUPING_SINGLE double points[3] = {x,y,z}
// #define POINT_ARG x, y, z

// extern "C" {
// Emulator<ND_TREE_EMULATOR_TYPE>* ND_TREE_EMULATOR_NAME_SETUP(const char* filename) {
//     return new Emulator<ND_TREE_EMULATOR_TYPE>(filename);
// }
// void ND_TREE_EMULATOR_NAME_INTERPOLATE(Emulator<ND_TREE_EMULATOR_TYPE>* emulator, double** points, size_t num_points, double* return_array) {
//     emulator->interpolate(points, num_points, return_array);
// }
// void ND_TREE_EMULATOR_NAME_FREE(Emulator<ND_TREE_EMULATOR_TYPE>* emulator){
//     free(emulator);
// }
// }

extern "C" {
   void ND_TREE_EMULATOR_NAME_SETUP(const char* filename, const char* mapping_filename, void*& emulator) {
       emulator = (void*)(new Emulator<ND_TREE_EMULATOR_TYPE>(filename, mapping_filename));
       // std::cout << "emulator: " << emulator << std::endl;
   }
   void ND_TREE_EMULATOR_NAME_INTERPOLATE(void*& emulator, POINT_INPUTS, size_t& num_points, double* return_array) {
       POINT_GROUPING;
       ((Emulator<ND_TREE_EMULATOR_TYPE> *)emulator)->interpolate(points, num_points, return_array);
   }
   void ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE(void*& emulator, POINT_INPUTS_SINGLE, double& return_array) {
       POINT_GROUPING_SINGLE;
       ((Emulator<ND_TREE_EMULATOR_TYPE> *)emulator)->interpolate(points, return_array);
   }
   void ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE_DX1(void*& emulator, POINT_INPUTS_SINGLE, double& return_array, double& dy_dx1) {
       POINT_GROUPING_SINGLE;
       ((Emulator<ND_TREE_EMULATOR_TYPE> *)emulator)->interpolate<1>(points, return_array, dy_dx1);
   }
   void ND_TREE_EMULATOR_NAME_FREE(void*& emulator){

       // std::cout << "emulator to delete: " << emulator << std::endl;
       delete((Emulator<ND_TREE_EMULATOR_TYPE>*)emulator);
   }
}