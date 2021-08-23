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
#include <omp.h>
//#include <morton-nd/mortonND_BMI2.h>
//#include <morton-nd/mortonND_LUT.h>
//
//using MortonND = mortonnd::MortonND;
#define POINT_CACHE_SIZE 4
#define CELL_CACHE_SIZE 100
#define INIT_CELL_CACHE_SIZE 128

template <typename indexing_int, size_t num_model_classes, size_t num_dim,
        size_t mapping_array_size, size_t encoding_array_size>
class Emulator {
public:
    /**
     * ND-Emulator. Loads a previously built/trained emulator.
     * @param filename Location of hdf5 file containing the emulator
     */
    Emulator(std::string filename, std::string mapping_location){
        std::string mphf_location = mapping_location;
        load_emulator(filename);
        // do domain transform
        for (size_t i = 0; i != num_dim; i++){
            domain_transform(&domain[i*2], i, 2);
            domain_transform(&index_domain[i*2], i, 2);
        }
        // compute dx
        for (size_t i = 0; i != num_dim; i++){
            dx[i] = (index_domain[i*2 + 1] - index_domain[i*2 + 0]) / (double)(1 << max_depth);
        }
        // compute other derived quantities
        weight_offset = 1<<num_dim; //std::pow(2, num_dim);
        num_cell_corners = weight_offset;
        // compute max index
        max_index = (1<<max_depth) - 1;
        // Compute weights for transforming cartesian point index into 1d dim
        for (size_t i = 0; i != num_dim; ++i) {
            point_index_transform_weights[i] = 1;
        }
        for (size_t i = 0; i < num_dim -1; i++){
            for (size_t j = i + 1; j < num_dim; j++){
                point_index_transform_weights[i] *= dims[j];
            }
        }
        // Compute weights for transforming cartesian cell index into 1d dim
        for (size_t i = 0; i != num_dim; ++i) {
            cell_index_transform_weights[i] = 1;
        }
        for (size_t i = 0; i < num_dim -1; i++){
            for (size_t j = i + 1; j < num_dim; j++){
                cell_index_transform_weights[i] *= dims[j] - 1;  // -1 since this is a cell index instead of a point index.
            }
        }

        // load minimal perfect hash function
        mphf = load_mphf(mphf_location);

        // allocate enough size for each thread to have its own storage
        weight_cache.resize(omp_get_max_threads()*weight_size);
    }

    void interpolate(const double* point, double& return_value){
        // get thread id
        size_t thread_id = omp_get_thread_num();
        // Check if we are already in correct cell.
        if (!point_in_current_cell(point, thread_id)){
            update_current_cell(point, thread_id);
        }
        return_value = nd_linear_interp(point, &(weight_cache.at(thread_id*weight_size)));
    }

    template<size_t derivative_dim>
    void interpolate(const double* point, double& return_value, double& derivative){
        // get thread id
        size_t thread_id = omp_get_thread_num();
        // Check if we are already in correct cell.
        if (!point_in_current_cell(point, thread_id)){
            update_current_cell(point, thread_id);
        }
        return_value = nd_linear_interp<derivative_dim>(point, &(weight_cache.at(thread_id*weight_size)), derivative);
    }


    /**
     * The interpolation is done using an emulator that has been computed offline and loaded when
     * the object is instantiated. The type of interpolation can vary throughout the table, which is
     * divided into cells. These cells are defined by a nd-tree decomposition. The mapping from the
     * input space to each cell is included in the offline emulator.
     * @param points An array of double pointers, with size num_dim.
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
     * @param num_points the number of points that will be interpolated over.
     * @param return_array Any array of size num_points that will be modified.
     */
    void interpolate(double** points, size_t num_points, double* return_array){
        // Setup cache
        std::vector<size_t> cached_cell_index;
        std::vector<unsigned short int> cached_types;
        std::vector<unsigned short int> cached_depths;
        std::vector<size_t> input_mapping(num_points);
        std::vector<double> local_cache;
        // -- reserve set size
        cached_cell_index.reserve(INIT_CELL_CACHE_SIZE);
        cached_types.reserve(INIT_CELL_CACHE_SIZE);
        cached_depths.reserve(INIT_CELL_CACHE_SIZE);
        local_cache.reserve(INIT_CELL_CACHE_SIZE*weight_size);



        // Determine needed cells
        // auto start = omp_get_wtime();
        #pragma omp parallel default(none) shared(cached_cell_index,cached_types,cached_depths, return_array,num_points, input_mapping, points, local_cache, node_values, weight_offset, domain ,dx)
        {
            // setup thread cache and determine number of threads and their chunk sizes
            std::vector<size_t> local_mapping;
            local_mapping.reserve(INIT_CELL_CACHE_SIZE);
            std::vector<size_t> local_depth;
            local_depth.reserve(INIT_CELL_CACHE_SIZE);
            std::vector<size_t> local_type;
            local_type.reserve(INIT_CELL_CACHE_SIZE);

            // Map input points to their corresponding cell
            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_points; ++i) {
                double point[num_dim];
                for (size_t j = 0; j < num_dim; ++j) {
                    point[j] = points[j][i];
                }
                unsigned short int depth;
                unsigned short int type;
                size_t cell_index = compute_cell_mapping(point, depth,  type);
                // determine if this is a new cell and save it if it is
                bool found = false;
                for (size_t j = 0; j < local_mapping.size(); ++j) {
                    if (local_mapping.at(j) == cell_index){
                        found = true;
                        input_mapping.at(i) = j;
                        break;
                    }
                }
                if (!found){
                    input_mapping.at(i) = local_mapping.size();
                    local_mapping.push_back(cell_index);
                    local_depth.push_back(depth);
                    local_type.push_back(type);
                }
            }
            // Create global list of needed cells
            #pragma omp critical
            {
                // We consolidate all the local unique cells into a list of global unique cells.
                // We change the local mapping array to hold the conversion to the local index to global index
                for (size_t i = 0; i < local_mapping.size(); ++i) {
                    bool found = false;
                    for (size_t j = 0; j < cached_cell_index.size(); ++j) {
                        if (cached_cell_index.at(j) == local_mapping.at(i)){
                            found = true;
                            local_mapping.at(i) = j;
                            break;
                        }
                    }
                    if (!found){
                        cached_cell_index.push_back(local_mapping.at(i));
                        cached_types.push_back(local_type.at(i));
                        cached_depths.push_back(local_depth.at(i));
                        local_mapping.at(i) = cached_cell_index.size() - 1;
                    }
                }
            }
            #pragma omp barrier
            #pragma omp single
            {
                // finish cache setup
                local_cache.resize(cached_cell_index.size()*weight_size);
            }

            // load needed cells
            #pragma omp for
            for (size_t i = 0; i < cached_cell_index.size(); ++i) {
                size_t local_cell_index[num_dim];
                un_global_indexing(cached_cell_index.at(i), local_cell_index);
                auto depth_diff = max_depth - cached_depths.at(i);
                for (size_t j = 0; j < (1<<num_dim); ++j) {
                    size_t point_index = compute_global_index_of_corner(local_cell_index, j, depth_diff);
                    local_cache.at(i*weight_size+j) = node_values[mphf(point_index)];
                }
                // Load domain info into weights
                auto cell_edge_index_size = 1 << depth_diff;
                for (size_t j = 0; j < num_dim; ++j) {
                    local_cache.at(i*weight_size+weight_offset+j) = domain[j*2]+dx[j]*local_cell_index[j];
                    local_cache.at(i*weight_size+weight_offset+num_dim+j) = domain[j*2]+dx[j]*(local_cell_index[j] + cell_edge_index_size);
                }
            }
            // Do interpolation on cells
            #pragma omp for schedule(static)
            for (size_t i = 0; i < num_points; ++i) {
                double point[num_dim];
                for (size_t j = 0; j < num_dim; ++j) {
                    point[j] = points[j][i];
                }
                size_t index = local_mapping.at(input_mapping.at(i));
                return_array[i] = nd_linear_interp(point, &(local_cache.at(weight_size*index)));//interp_point(point, &(local_cache.at(weight_size*input_mapping.at(i))));
                // return_array[i] = nd_linear_interp(point, &(local_cache.at(weight_size*input_mapping.at(i))));//interp_point(point, &(local_cache.at(weight_size*input_mapping.at(i))));
            }
        }
    }

private:
    std::vector<double> weight_cache;
    size_t weight_size = (1<<num_dim)+num_dim*2;
    size_t max_index;
    size_t max_depth;
    size_t weight_offset;
    size_t num_cell_corners;
    size_t model_classes[num_model_classes];
    size_t transforms[num_model_classes];
    size_t spacing[num_dim];
    size_t dims[num_dim];
    size_t point_index_transform_weights[num_dim];
    size_t cell_index_transform_weights[num_dim];
    double dx[num_dim];
    double domain[num_dim * 2];
    double index_domain[num_dim * 2];
    char encoding_array[encoding_array_size];
    double node_values[mapping_array_size];
    size_t current_model_type_index;
    /* Declare the PTHash function. */
    typedef pthash::single_phf<pthash::murmurhash2_64,         // base hasher
            pthash::compact_compact,  // encoder type
            true                    // minimal
    > pthash_type;
    pthash_type mphf;

    void update_current_cell(const double* input_point, size_t i){
        // determine index
        unsigned short int depth;
        unsigned short int type;
        size_t cell_indices[num_dim];
        size_t cell_index = compute_cell_mapping(input_point, depth,  type, cell_indices);

        // determine which cell it is in
        auto depth_diff = max_depth - depth;
        for (size_t j = 0; j < (1<<num_dim); ++j) {
            size_t point_index = compute_global_index_of_corner(cell_indices, j, depth_diff);
            weight_cache.at(i*weight_size+j) = node_values[mphf(point_index)];
        }
        // Load domain info into weights
        auto cell_edge_index_size = 1 << depth_diff;
        for (size_t j = 0; j < num_dim; ++j) {
            weight_cache.at(i*weight_size+weight_offset+j) = domain[j*2]+dx[j]*cell_indices[j];
            weight_cache.at(i*weight_size+weight_offset+num_dim+j) = domain[j*2]+dx[j]*(cell_indices[j] + cell_edge_index_size);
        }

    }

    bool point_in_current_cell(const double* point_domain, size_t thread_id){
        const double* current_cell_domain = get_cell_domain(&(weight_cache.at(thread_id*weight_size)));
        // determine if point is in current domain.
        for (size_t i = 0; i<num_dim; i++){
            if ((point_domain[i] < current_cell_domain[i])
            || (point_domain[i] > current_cell_domain[i+num_dim])){
                return false;
            }
        }
        return true;
    }

    void compute_cartesian_indices(const double *point, size_t *cartesian_indices) const {
        for (size_t i = 0; i < num_dim; i++){
            // restrict the point to fall within the real domain (as opposed to the index domain)
            double p = std::max(std::min(point[i], domain[i * 2 + 1]), domain[i * 2 + 0]);
            // compute cartesian index
            cartesian_indices[i] = static_cast<size_t>((p - domain[i * 2 + 0]) / dx[i]);
            // If the index is outside the index domain of the emulator round to the nearest cell.
            cartesian_indices[i] = std::max(static_cast<size_t>(0), cartesian_indices[i]);
            cartesian_indices[i] = std::min(max_index, cartesian_indices[i]);
        }
    }

    size_t compute_global_index(const size_t* cell_indices){
        size_t global_index = 0;
        for (unsigned int i = 0; i < num_dim; ++i) {
            global_index += cell_indices[i]*(cell_index_transform_weights[i]);
        }
        return global_index;
    }

    size_t trimmed_and_compute_global_index(size_t* cell_indices, const unsigned short int depth_diff){
        size_t global_index = 0;
        for (unsigned int i = 0; i < num_dim; ++i) {
            cell_indices[i] = ((cell_indices[i]>>depth_diff)<<depth_diff);
            global_index += cell_indices[i]*(cell_index_transform_weights[i]);
        }
        return global_index;
    }

    size_t compute_global_index_of_corner(const size_t* cell_indices, unsigned int corner,
                                                 unsigned short int depth_diff){
        size_t cell_edge_size = (1<<depth_diff);
        size_t global_index = 0;
        for (unsigned int i = 0; i < num_dim; ++i) {
            if ((corner >> i) & 1){
                global_index += (cell_indices[i] + cell_edge_size)*(point_index_transform_weights[i]);
            } else{
                global_index += cell_indices[i]*(point_index_transform_weights[i]);
            }
        }
        return global_index;
    }

    pthash_type load_mphf(std::string location){
        /* Set up a build configuration. */
        pthash::build_configuration config;
        config.c = 2.5;
        config.alpha = 0.99;
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

    void un_global_indexing(size_t global_index, size_t local_index[num_dim]){
        for (unsigned int i = 0; i < num_dim; ++i) {
            local_index[i] = global_index / cell_index_transform_weights[i];
            global_index = global_index % cell_index_transform_weights[i];
        }
    }

    // compute the depth, type, global_index and cell_index for the given input_point
    size_t compute_cell_mapping(const double* input_point, unsigned short int& depth, unsigned short int& type, size_t* cell_index){
        // Compute index
        compute_cartesian_indices(input_point, cell_index);
        // unpack depth and type from char array
        // -- compute global index for local cell
        size_t global_index = compute_global_index(cell_index);
        // -- grab encoded byte and decode
        char byte = encoding_array[global_index];
        depth = byte & 0b00001111;
        type  = byte >> 4;
        // determine lower corner of the cell that contains this cell (this takes into account
        // that the cell found before is the smallest resolved cell, so if the depth is not the max
        // depth it will be a subset of the cell we want to load).
        auto depth_diff = max_depth - depth;
        global_index = trimmed_and_compute_global_index(cell_index, depth_diff);
        return global_index;
    }

    // compute the depth, type, global_index and cell_index for the given input_point
    size_t compute_cell_mapping(const double* input_point, unsigned short int& depth, unsigned short int& type){
        size_t cell_index[num_dim];
        return compute_cell_mapping(input_point, depth, type, cell_index);
    }

    void domain_transform(double* dim_array, size_t dim, size_t num_vars){
        if (spacing[dim] == 0){
            return;
        } else if (spacing[dim] == 1){
            for (size_t j = 0; j != num_vars; j++){
                dim_array[j] = log10(dim_array[j]);
            }
        } else {
            throw std::out_of_range ("No valid spacing implemented");
        }
    }

    void load_emulator(const std::string& file_location){
        // Load hdf5 file
        std::cout << file_location << std::endl;
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
        dataset = mapping_group.getDataSet("node_values_encoded");
        dataset.template read(node_values);
    }

    const double* get_cell_domain(const double* weights){
        // return pointer to the domain
        return &weights[(1<<num_dim)];
    }

    double interp_point(const double* point, const double* current_weights){;
        // Choose which interpolation scheme to use
        if (model_classes[current_model_type_index] == MODEL_CLASS_TYPE_ND_LINEAR){
            if (transforms[current_model_type_index] == TRANSFORMS_NONE){
                return nd_linear_interp(point, current_weights);
            }
            // TODO: allow for multiple model classes
//            else if (transforms[current_model_type_index] == TRANSFORMS_LOG){
//                double solution = nd_linear_interp(point, current_weights+1);
//                solution = pow(10.0, solution);
//                if (current_weights[0] != 0.0){
//                    solution -= std::fabs(current_weights[0]);
//                    solution *=  copysign(1.0, current_weights[0]);
//                }
//                return solution;
//            }
            else{
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
        const double* cell_domain = get_cell_domain(weight);
        #pragma omp simd
        for (size_t i = 0; i < num_dim; i++){
            x[i] = (point[i] - cell_domain[i]) / (cell_domain[num_dim + i] - cell_domain[i]);
        }
        // apply weights
        double solution = 0;
        #pragma omp simd reduction(+:solution)
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

    template<size_t derivative_dimension>
    double nd_linear_interp(const double* point, const double* weight, double& dydx0){
        /*
         * do an ND-linear interpolation at the points in X given model weight values.
         * https://math.stackexchange.com/a/1342377
         *
         */
        // transform point to domain [0,1]
        double x[num_dim];
        const double* cell_domain = get_cell_domain(weight);
        #pragma omp simd
        for (size_t i = 0; i < num_dim; i++){
            x[i] = (point[i] - cell_domain[i]) / (cell_domain[num_dim + i] - cell_domain[i]);
        }
        // apply weights
        double solution = 0;
        dydx0 = 0;
        #pragma omp simd reduction(+:solution,dydx0)
        for (size_t i = 0; i < weight_offset; i++) {
            double w = 1;
            for (size_t j = 0; (j < derivative_dimension); j++) {  //&& (j != derivative_dimension)
                auto bit = (i >> j) & 1;
                w *= bit == 0 ? (1 - x[j]) : x[j];
            }
            for (size_t j = derivative_dimension+1; (j < num_dim); j++) {  //&& (j != derivative_dimension)
                auto bit = (i >> j) & 1;
                w *= bit == 0 ? (1 - x[j]) : x[j];
            }
            auto bit = (i >> derivative_dimension) & 1;
            dydx0 += weight[i] * w * ( bit == 0 ? -1 : 1);
            solution += weight[i] * w * (bit == 0 ? (1 - x[derivative_dimension]) : x[derivative_dimension]);
        }
        // transform derivative back
        dydx0 /= static_cast<double>(cell_domain[num_dim + derivative_dimension] - cell_domain[derivative_dimension]);
        return solution;
    }

};

#endif //CPP_EMULATOR_EMULATOR_H
