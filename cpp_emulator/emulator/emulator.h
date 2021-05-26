//
// Created by jared on 5/25/2021.
//

#ifndef CPP_EMULATOR_EMULATOR_H
#define CPP_EMULATOR_EMULATOR_H
#include "string"
#include "H5Cpp.h"

template <class encoding_int, class indexing_int>
class Emulator {
public:
    Emulator(std::string filename){
        std::cout << "Made new object!";
    }
private:
    int num_dim;
    int num_model_classes;
    int num_models;
    int* offsets;
    encoding_int * encoding_array;
    indexing_int * indexing_array;
    double ** mapping_arrays;
};



#endif //CPP_EMULATOR_EMULATOR_H
