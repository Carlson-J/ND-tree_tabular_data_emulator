//
// Created by jared on 5/25/2021.
//

#ifndef CPP_EMULATOR_EMULATOR_H
#define CPP_EMULATOR_EMULATOR_H

#include <highfive/H5File.hpp>
#include "string"
#include "H5Cpp.h"
#include "constants.h"

template <class encoding_int, class indexing_int>
class Emulator {
public:
    Emulator(std::string filename){
        std::cout << "Loading emulator" << std::endl;
        load_emulator(filename);
    }
private:
    int num_dim;
    int num_model_classes;
    int num_models;
    int * offsets;
    encoding_int * encoding_array;
    indexing_int * indexing_array;
    double ** mapping_arrays;

    void load_emulator(const std::string& file_location){
        // Load hdf5 file
        HighFive::File file(file_location, HighFive::File::ReadOnly);

        // Load the attributes of the emulator
        HighFive::Attribute attribute = file.getAttribute("dims");
        // -- dims
        std::vector<int> dims;
        attribute.template read(dims);
        // -- domain
        attribute = file.getAttribute("domain");
        std::vector<std::vector<double>> domain;
        attribute.template read(domain);
        // -- error threshold
        double error_threshold;
        attribute = file.getAttribute("error_threshold");
        attribute.template read(error_threshold);
        // -- max depth
        int max_depth;
        attribute = file.getAttribute("max_depth");
        attribute.template read(max_depth);
        // -- max test points
        int max_test_points;
        attribute = file.getAttribute("max_test_points");
        attribute.template read(max_test_points);
        // -- relative error
        double relative_error;
        attribute = file.getAttribute("relative_error");
        attribute.template read(relative_error);
        // -- model clases
        attribute = file.getAttribute("model_classes");
        num_model_classes = attribute.getSpace().getDimensions()[0];
        HighFive::FixedLenStringArray<100> model_classes;
        if (num_model_classes == 1){
            attribute.template read(model_classes[0]);
        } else{
            attribute.template read(model_classes);
        }

//        // -- model clases
//        std::vector<std::string> model_classes;
//        attribute = file.getAttribute("model_classes");
//        attribute.template read(model_classes);

        for(auto s : model_classes)
            std::cout << "model classes : " << s << std::endl;

        std::cout << "error_threshold " << error_threshold << std::endl;
        std::cout << "max_depth " << max_depth << std::endl;
        std::cout << "max_test_points " << max_test_points << std::endl;
        std::cout << "relative_error " << relative_error << std::endl;

        // print attribute
        for (const auto v : domain) {
            for (const auto a : v) {
                std::cout << "domain: " << a << std::endl;
            }
        }
        std::cout << error_threshold << std::endl;
    }
};



#endif //CPP_EMULATOR_EMULATOR_H
