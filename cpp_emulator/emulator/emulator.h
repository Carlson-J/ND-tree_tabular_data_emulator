//
// Created by jared on 5/25/2021.
//

#ifndef CPP_EMULATOR_EMULATOR_H
#define CPP_EMULATOR_EMULATOR_H

#include <highfive/H5File.hpp>
#include "string"
#include "H5Cpp.h"
#include "constants.h"
#include <memory>

template <class encoding_int, class indexing_int>
class Emulator {
public:
    Emulator(std::string filename){
        std::cout << "Loading emulator" << std::endl;
        load_emulator(filename);
    }
private:
    size_t num_dim;
    size_t num_model_classes;
    size_t num_models;
    std::vector<encoding_int> offsets;
    std::vector<encoding_int> encoding_array;
    std::vector<encoding_int> indexing_array;
    std::vector<std::vector<std::vector<double>>> model_arrays;

    void load_emulator(const std::string& file_location) {
        // Load hdf5 file
        HighFive::File file(file_location, HighFive::File::ReadOnly);

        // Load the attributes of the emulator
        HighFive::Attribute attribute = file.getAttribute("dims");
        // -- dims
        std::vector<int> dims;
        attribute.template read(dims);
        this->num_dim = dims.size();
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
        // -- model classes
        std::vector<int> model_classes;
        attribute = file.getAttribute("model_classes");
        attribute.template read(model_classes);
        // -- spacings
        std::vector<int> spacings;
        attribute = file.getAttribute("spacing");
        attribute.template read(spacings);

        // Load mapping arrays
        HighFive::Group mapping_group = file.getGroup("mapping");
        // -- encoding array
        HighFive::DataSet dataset = mapping_group.getDataSet("encoding");
        dataset.template read(encoding_array);
        // -- indexing array
        dataset = mapping_group.getDataSet("indexing");
        dataset.template read(indexing_array);
        // -- offsets array
        dataset = mapping_group.getDataSet("offsets");
        dataset.template read(offsets);
        this->num_models = encoding_array.size();

        // Load model arrays
        HighFive::Group model_group = file.getGroup("models");
        // -- Create temp array to store it in
        auto model_types = model_group.listObjectNames();
        this->num_model_classes = model_types.size();
        for (const auto &model_name : model_types) {
            dataset = model_group.getDataSet(model_name);
            std::vector<std::vector<double>> model_data;
            dataset.template read(model_data);
            model_arrays.push_back(model_data);
        }
    }
};



#endif //CPP_EMULATOR_EMULATOR_H
