//
// Created by jared on 5/25/2021.
//

#ifndef CPP_EMULATOR_EMULATOR_H
#define CPP_EMULATOR_EMULATOR_H

#include "../HighFive/include/highfive/H5File.hpp"
#include "../pthash/include/pthash.hpp"
#include <string>
#include "H5Cpp.h"
#include "constants.h"
#include <memory>
#include <algorithm>
#include <cmath>
//#include <morton-nd/mortonND_BMI2.h>
//#include <morton-nd/mortonND_LUT.h>
//
//using MortonND = mortonnd::MortonND;

template <typename indexing_int, size_t num_model_classes, size_t num_dim,
        size_t mapping_array_size, size_t encoding_array_size>
class Emulator {
public:
    Emulator(std::string filename, std::string mphf_location="./pthash.bin"){
        std::cout << "Loading emulator" << std::endl;
        load_emulator(filename);
        // do domain transform
        for (size_t i = 0; i < num_dim; i++){
            domain_transform(&domain[i*2], i, 2);
            domain_transform(&index_domain[i*2], i, 2);
        }
        // compute dx
        for (size_t i = 0; i < num_dim; i++){
            dx[i] = (index_domain[i*2 + 1] - index_domain[i*2 + 0]) / (double)(1 << max_depth);
        }
        // compute other derived quantities
        weight_offset = 1<<num_dim; //std::pow(2, num_dim);
        num_cell_corners = weight_offset;
        // compute max index
        max_index = (1<<max_depth) - 1; //size_t(pow(2, max_depth) - 1);
        // Compute weights for transforming cartesian into 1d dim
        for (int i = 0; i < num_dim; ++i) {
            index_transform_weights[i] = 1;
        }
        for (int i = num_dim-2; i >= 0; --i) {
            for (int j = i; j >= 0; --j) {
                index_transform_weights[j] *= dims[i];
            }
        }
        // load minimal perfect hash function
        mphf = load_mphf(mphf_location);
        // Set current cell to the first cell, morton index 0, by giving the lower part of the domain
        double starting_cell[num_dim];
        for (size_t i = 0; i < num_dim; i++){
            starting_cell[i] = domain[i*2];
        }
        update_current_cell(starting_cell);
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
         * the object is instantiated. The type of interpolation can vary throughout the table, which is
         * divided into cells. These cells are defined by a nd-tree decomposition. The mapping from the
         * input space to each cell is included in the offline emulator.
         */
        // Determine which model (i.e. interpolator) each point will use.
        for (size_t i = 0; i < num_points; i++){
            double point_domain[num_dim];
            double point[num_dim];
            for (size_t j = 0; j < num_dim; j++){
                point_domain[j] = points[j][i];
                point[j] = points[j][i];
                // Do any needed domain transforms
                domain_transform(&point_domain[j], j, 1);
            }
            // Check if we are already in correct cell.
            if (!point_in_current_cell(point_domain)){
                update_current_cell(point_domain);
            }
            return_array[i] = interp_point(point);
        }
    }

//    encoding_int compute_tree_index(const double* point){
//        // Compute index of the cell that the point falls in in the tree index space
//        size_t cartesian_indices[num_dim];
//        compute_cartesian_indices(point, cartesian_indices);
//        // TODO: change this to use nd-morton
//        // convert to tree index space
//        size_t index = 0;
//        for (size_t i = 0; i < max_depth; i++){
//            for (size_t j = 0; j < num_dim; j++){
//                index = (index << 1) | ((cartesian_indices[num_dim - 1 - j] >> (max_depth - i - 1)) & 1);
//            }
//        }
//        return encoding_int(index);
//    }

    void compute_cartesian_indices(const double *point, size_t *cartesian_indices) const {
        for (size_t i = 0; i < num_dim; i++){
            // restrict the point to fall within the real domain (as opposed to the index domain)
            double p = std::max(std::min(point[i], domain[i * 2 + 1]), domain[i * 2 + 0]);
            // compute cartesian index
            cartesian_indices[i] = size_t((p - domain[i * 2 + 0]) / dx[i]);
            // If the index is outside the index domain of the emulator round to the nearest cell.
            cartesian_indices[i] = std::max(size_t(0), cartesian_indices[i]);
            cartesian_indices[i] = std::min(max_index, cartesian_indices[i]);
        }
    }

private:
    size_t max_index;
    size_t max_depth;
    size_t weight_offset;
    size_t num_cell_corners;
    size_t model_classes[num_model_classes];
    size_t transforms[num_model_classes];
    size_t model_class_weights[num_model_classes];
    size_t spacing[num_dim];
    size_t dims[num_dim];
    size_t index_transform_weights[num_dim];
    double dx[num_dim];
    double domain[num_dim * 2];
    double index_domain[num_dim * 2];
//    size_t offsets[num_model_classes];
//    size_t model_array_offsets[num_model_classes];
    char encoding_array[encoding_array_size];
    double node_values[mapping_array_size];
//    double model_arrays[model_array_size];
    double* current_cell_domain;
    double current_weights[50];
    size_t current_model_type_index;
    /* Declare the PTHash function. */
    typedef pthash::single_phf<pthash::murmurhash2_64,         // base hasher
            pthash::dictionary_dictionary,  // encoder type
            true                    // minimal
    > pthash_type;
    pthash_type mphf;

    pthash_type load_mphf(std::string location){
        /* Set up a build configuration. */
        pthash::build_configuration config;
        config.c = 6.0;
        config.alpha = 0.94;
        config.minimal_output = true;  // mphf
        config.verbose_output = true;

        // config.num_partitions = 50;
        // config.num_threads = 4;
        // typedef partitioned_mphf<murmurhash2_64,        // base hasher
        //                          dictionary_dictionary  // encoder type
        //                          >
        //     pthash_type;

        // Load pthash
        pthash_type f;
        essentials::load(f, location.c_str());
        return f;
    }

    // compute the depth, type, and cell index for the given input_point
    void compute_cell_mapping(double* input_point, size_t& depth, size_t& type, size_t* cell_index){
        // Compute index
        compute_cartesian_indices(input_point, cell_index);
        // unpack depth and type from char array
        // -- compute global index
        size_t global_index = 0;
        for (unsigned int i = 0; i < num_dim; ++i) {
            global_index += cell_index[i]*(index_transform_weights[i] - 1); // -1 since this is a cell index instead of a point index.
        }
        // -- grab encoded byte and decode
        char byte = encoding_array[global_index];
        depth = byte & 0b00001111;
        type  = byte >> 4;
    }

    void update_current_cell(double* point_domain){
        // Determine type and depth of cell
        size_t depth = 0;
        size_t type = 0;
        size_t cell_index[num_dim];
        compute_cell_mapping(point_domain, depth, type, cell_index);
        auto depth_diff = max_depth - depth;
        auto cell_edge_index_size = 1 << depth_diff;
        // determine node points needed
        for (unsigned int i = 0; i != num_cell_corners; ++i) {
            size_t corner_index[num_dim];
            // determine corner index
            for (int j = 0; j != num_dim; ++j) {
                corner_index[j] = cell_index[j];
                if ((i >> j) & 1){
                    corner_index[j] += cell_edge_index_size;
                }
            }
            // compute global index
            double global_index = 0;
            for (int j = 0; j < num_dim; ++j) {
                global_index += corner_index[j]*(index_transform_weights[j]);
            }
            // Compute hash and load value
            current_weights[i] = node_values[mphf(global_index)];
        }
        // save domain information
        // lower corner
        for (int i = 0; i < num_dim; ++i) {
            current_weights[weight_offset+i] = domain[i*2]+dx[i]*cell_index[i];
        }
        // upper corner
        for (int i = 0; i < num_dim; ++i) {
            current_weights[weight_offset+num_dim+i] = domain[i*2]+dx[i]*(cell_index[i] + cell_edge_index_size);
        }
        current_cell_domain = compute_cell_domain(current_weights);
    }

    bool point_in_current_cell(const double* point_domain){
        // determine if point is in current domain.
        for (size_t i = 0; i<num_dim; i++){
            if ((point_domain[i] < current_cell_domain[i])
                || (point_domain[i] > current_cell_domain[i+num_dim])){
                return false;
            }
        }
        return true;
    }

    void domain_transform(double* dim_array, size_t dim, size_t num_vars){
        if (spacing[dim] == 0){
            return;
        } else if (spacing[dim] == 1){
            for (size_t j = 0; j < num_vars; j++){
                dim_array[j] = log10(dim_array[j]);
            }
        } else {
            throw std::out_of_range ("No valid spacing implemented");
        }
    }

    void load_emulator(const std::string& file_location){
        // Load hdf5 file
        HighFive::File file(file_location, HighFive::File::ReadOnly);

        // -- domain
        HighFive::Attribute attribute = file.getAttribute("domain");
        attribute.template read(domain);
        assert(num_dim*2 == attribute.getSpace().getElementCount());
        // -- index domain
        attribute = file.getAttribute("index_domain");
        attribute.template read(index_domain);
        assert(num_dim*2 == attribute.getSpace().getElementCount());
        // -- max depth
        attribute = file.getAttribute("max_depth");
        attribute.template read(max_depth);
        // -- model classes
        attribute = file.getAttribute("model_classes");
        attribute.template read(model_classes);
        assert(num_model_classes == attribute.getSpace().getElementCount());
        // -- transforms
        attribute = file.getAttribute("transforms");
        attribute.template read(transforms);
        assert(num_model_classes == attribute.getSpace().getElementCount());
        // -- spacings
        attribute = file.getAttribute("spacing");
        attribute.template read(spacing);
        assert(num_dim == attribute.getSpace().getElementCount());
        // -- dims array
        attribute = file.getAttribute("dims");
        attribute.template read(dims);
        assert(num_dim == attribute.getSpace().getElementCount());

        // Load mapping arrays
        HighFive::Group mapping_group = file.getGroup("mapping");
        // -- encoding array
        HighFive::DataSet dataset = mapping_group.getDataSet("encoding");
        dataset.template read(encoding_array);
        dataset = mapping_group.getDataSet("node_values");
        dataset.template read(node_values);
//        assert(num_models == dataset.getElementCount());
//        // -- indexing array
//        dataset = mapping_group.getDataSet("indexing");
//        dataset.template read(indexing_array);
//        assert(num_models == dataset.getElementCount());
//        // -- offsets array
//        dataset = mapping_group.getDataSet("offsets");
//        dataset.template read(offsets);
//        assert(num_model_classes == dataset.getElementCount());


//        // Load model arrays
//        HighFive::Group model_group = file.getGroup("models");
//        auto model_types = model_group.listObjectNames();
//        size_t current_offset = 0;
//        for (size_t i = 0; i < num_model_classes; i++) {
//            model_array_offsets[i] = current_offset;
//            dataset = model_group.getDataSet(model_types[i]);
//            dataset.template read(&(model_arrays[current_offset]));
//            current_offset += dataset.getElementCount();
//            model_class_weights[i] = dataset.getDimensions()[1];
//        }
//        assert(current_offset == model_array_size);
//        size_t mapping_size = 0;
//        for (auto& size : dims){
//            mapping_size += size;
//        }
//        assert(mapping_array_size == mapping_size);
    }

//    indexing_int get_model_index(const double* point){
//        /*
//         * point: point in num_dim space.
//         *
//         * This function finds the corresponding model that should do the interpolation of this point.
//         */
//        // compute tree index
//        encoding_int tree_index = compute_tree_index(point);
//        // decode index
//        auto start = std::begin(encoding_array);
//        auto end = std::end(encoding_array);
//        auto index = std::upper_bound(start, end, tree_index) - start;
//        return indexing_array[index];
//    }
//
//
//    double* get_weights(const size_t index){
//        // get model weights
//        return &model_arrays[model_array_offsets[current_model_type_index]
//                                + (index - offsets[current_model_type_index])
//                                *model_class_weights[current_model_type_index]];
//    }

    double* compute_cell_domain(double* weights){
        // return pointer to the domain
        return &weights[weight_offset];
    }


    double interp_point(const double* point){;
        // Choose which interpolation scheme to use
        if (model_classes[current_model_type_index] == MODEL_CLASS_TYPE_ND_LINEAR){
            if (transforms[current_model_type_index] == TRANSFORMS_NONE){
                return nd_linear_interp(point, current_weights);
            } else if (transforms[current_model_type_index] == TRANSFORMS_LOG){
                double solution = nd_linear_interp(point, current_weights+1);
                solution = pow(10.0, solution);
                if (current_weights[0] != 0.0){
                    solution -= std::fabs(current_weights[0]);
                    solution *=  copysign(1.0, current_weights[0]);
                }
                return solution;
            } else{
                throw std::exception(); //"Unknown transform"
            }
        } else{
            throw std::exception(); //"Model class not implemented yet"
        }
    }

    // ------ Model Classes ------ //
    double nd_linear_interp(const double* point, const double* weight){
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
