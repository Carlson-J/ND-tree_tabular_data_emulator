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
#include <algorithm>
#include <cmath>

template <class encoding_int, class indexing_int>
class Emulator {
public:
    Emulator(std::string filename){
        std::cout << "Loading emulator" << std::endl;
        load_emulator(filename);
        // compute dx
        for (size_t i = 0; i < num_dim; i++){
            dx.push_back((domain[i][1] - domain[i][0]) / (dims[i] - 1));
        }
        // compute other derived quantities
        weight_offset = std::pow(2, num_dim);
    }

    void interpolate(double** points, size_t num_points, double* return_array){
        /*
         * points: An array of double pointers, with size num_dim.
         *      Each pointer points to an array of doubles size num_points, with the index
         *      corresponding dimension of the point, i.e.,
         *      points = [
         *          [x0_0, x0_1, ...,x0_num_points],
         *          [x1_0, x1_1, ...,x1_num_points],
         *          ...
         *          [xN_0, xN_1, ...,xN_num_points],
         *      ]
         *      where N = num_dim.
         *      This allows for any number of dimensions to be used without changing the method.
         * num_points: the number of points that will be interpolated over.
         * return_array: Any array of size num_points that will be modified.
         *
         * The interpolation is done using an emulator that has been computed offline and loaded when
         * the object is instantiated. The type of interpolation can vary throught the table, which is
         * divided into cells. These cells are defined by a nd-tree decomposition. The mapping from the
         * input space to each cell is included in the offline emulator.
         */
        // Determine which model (i.e. interpolator) each point will use.
        size_t mapping_array[num_points];
        for (size_t i = 0; i < num_points; i++){
            double point[num_dim];
            for (size_t j = 0; j < num_dim; j++){
                point[j] = points[j][i];
            }
            mapping_array[i] = get_model_index(point);
        }
        // Do interpolation
        for (size_t i = 0; i < num_points; i++){
            double point[num_dim];
            for (size_t j = 0; j < num_dim; j++){
                point[j] = points[j][i];
            }
            return_array[i] = interp_point(point, mapping_array[i]);
        }


    }

private:
    size_t num_dim;
    size_t num_model_classes;
    size_t num_models;
    size_t max_depth;
    size_t weight_offset;
    std::vector<size_t> model_classes;
    std::vector<double> dx;
    std::vector<size_t> dims;
    std::vector<std::vector<double>> domain;
    std::vector<encoding_int> offsets;
    std::vector<encoding_int> encoding_array;
    std::vector<indexing_int> indexing_array;
    std::vector<std::vector<std::vector<double>>> model_arrays;

    void load_emulator(const std::string& file_location) {
        // Load hdf5 file
        HighFive::File file(file_location, HighFive::File::ReadOnly);

        // Load the attributes of the emulator
        HighFive::Attribute attribute = file.getAttribute("dims");
        // -- dims
        attribute.template read(dims);
        num_dim = dims.size();
        // -- domain
        attribute = file.getAttribute("domain");
        attribute.template read(domain);
        // -- error threshold
        double error_threshold;
        attribute = file.getAttribute("error_threshold");
        attribute.template read(error_threshold);
        // -- max depth
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
        num_models = encoding_array.size();

        // Load model arrays
        HighFive::Group model_group = file.getGroup("models");
        // -- Create temp array to store it in
        auto model_types = model_group.listObjectNames();
        num_model_classes = model_types.size();
        for (const auto &model_name : model_types) {
            dataset = model_group.getDataSet(model_name);
            std::vector<std::vector<double>> model_data;
            dataset.template read(model_data);
            model_arrays.push_back(model_data);
        }
    }

    size_t get_model_index(const double* point){
        /*
         * point: point in num_dim space.
         *
         * This function finds the corresponding model that should do the interpolation of this point.
         */
        // compute tree index
        encoding_int tree_index = compute_tree_index(point);
        // decode index
        auto start = encoding_array.begin();
        auto end = encoding_array.end();
        size_t index = std::upper_bound(start, end, tree_index) - start;
        index = indexing_array[index];
        // return index
        return index;
    }

    encoding_int compute_tree_index(const double* point){
        // Compute index of the cell that the point falls in in the tree index space
        size_t cartesian_index[num_dim];
        for (size_t i = 0; i < num_dim; i++){
            cartesian_index[i] = size_t((point[i] - domain[i][0])/dx[i]);
            // If the index is outside the domain of the emulator round to the nearest cell.
            cartesian_index[i] = std::max(size_t(0), cartesian_index[i]);
            cartesian_index[i] = std::min(size_t(dims[i] - 2), cartesian_index[i]);
        }
        // convert to tree index space
        size_t index = 0;
        for (size_t i = 0; i < max_depth; i++){
            for (size_t j = 0; j < num_dim; j++){
                index = (index << 1) | ((cartesian_index[num_dim - j] >> (max_depth - i - 1)) & 1);
            }
        }
        return encoding_int(index);
    }

    double interp_point(const double* point, const size_t index){
        // Determine which model array to use
        size_t model_type_index = std::upper_bound(offsets.begin(), offsets.end(), index) - offsets.begin() - 1;
        // get model weights
        std::vector<double>& weights = model_arrays[model_type_index][index - offsets[model_type_index]];
        // Choose which interpolation scheme to use
        if (model_classes[model_type_index] == MODEL_CLASS_TYPE_ND_LINEAR){
            return nd_linear_interp(point, weights);
        } else{
            throw std::exception(), "Model class not implemented yet";
        }
    }

    // ------ Model Classes ------ //
    double nd_linear_interp(const double* point, const std::vector<double>& weight){
        /*
         * do an ND-linear interpolation at the points in X given model weight values.
         * https://math.stackexchange.com/a/1342377
         *
         */
        // transform point to domain [0,1]
        double x[num_dim];
        for (size_t i = 0; i < num_dim; i++){
            x[i] = (point[i] - weight[weight_offset + i]) / (weight[weight_offset + num_dim + i] - weight[weight_offset + i]);
        }
        // apply weights
        double solution = 0;
        for (size_t i = 0; i < weight_offset; i++){
            double w = 1;
            for (size_t j = 0; j < num_dim; j++){
                auto bit = (i >> j) & 1;
                w *= bit == 0 ? (1 - x[j]) : x[j];
            }
            solution += weight[i] * w;
        }
        return solution;
    }
};



#endif //CPP_EMULATOR_EMULATOR_H
