//
// Created by jared on 5/31/2021.
//
#include "emulator.h"
#include "table_params.h"

extern "C" {
    Emulator<ND_TREE_EMULATOR_TYPE>* ND_TREE_EMULATOR_NAME_SETUP(const char* filename) {
        return new Emulator<ND_TREE_EMULATOR_TYPE>(filename);
    }
    void ND_TREE_EMULATOR_NAME_INTERPOLATE(Emulator<ND_TREE_EMULATOR_TYPE>* emulator, double** points, size_t num_points, double* return_array) {
        emulator->interpolate(points, num_points, return_array);
    }
    void ND_TREE_EMULATOR_NAME_FREE(Emulator<ND_TREE_EMULATOR_TYPE>* emulator){
        free(emulator);
    }
}