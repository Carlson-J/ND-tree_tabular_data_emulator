//
// Created by jared on 5/31/2021.
//
#include "emulator.h"
#include "../Tests/saved_emulator_4d_constants.h"
extern "C" {
int print_me(const char* filename) {
    std::cout << "testing" << std::endl;
    std::printf("%s", filename);
    std::cout << std::endl << std::endl;
    return 6;}
void send_array(int* ints, size_t num_ints){
    for (int i = 0; i < num_ints; i++){
        std::cout << ints[i] << std::endl;
        ints[i]++;
    }
}
Emulator<ND_TREE_EMULATOR_TYPE>* setup_emulator(const char* filename) {
    std::cout << filename << std::endl;
    return new Emulator<ND_TREE_EMULATOR_TYPE>(filename);
}
void interpolate(Emulator<ND_TREE_EMULATOR_TYPE>* emulator, double** points, size_t num_points, double* return_array) {
//    for (int i = 0; i < num_points; i++){
//        std::cout << return_array[i] << std::endl;
//        return_array[i] = 99;
//    }
    emulator->interpolate(points, num_points, return_array);
}
}