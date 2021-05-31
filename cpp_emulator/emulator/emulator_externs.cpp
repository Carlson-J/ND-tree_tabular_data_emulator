//
// Created by jared on 5/31/2021.
//
#include "emulator.h"
#include "../Tests/saved_emulator_4d_constants.h"
extern "C" {
    Emulator<ND_TREE_EMULATOR_TYPE>* setup_emulator(const char* filename) {
        std::cout << filename << std::endl;
        return new Emulator<ND_TREE_EMULATOR_TYPE>(filename);
    }
    void interpolate(Emulator<ND_TREE_EMULATOR_TYPE>* emulator, double** points, size_t num_points, double* return_array) {
        emulator->interpolate(points, num_points, return_array);
    }
    void free_emulator(Emulator<ND_TREE_EMULATOR_TYPE>* emulator){
        free(emulator);
    }
}