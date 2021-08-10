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
#include <mutex>
#include <shared_mutex>
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
    Emulator(std::string filename){
        std::string mphf_location = "./pthash.bin";
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
        std::vector<size_t> local_cache_cell_index;
        std::vector<unsigned short int> local_cache_types;
        std::vector<unsigned short int> local_cache_depths;
        std::vector<size_t> input_mapping(num_points);
        std::vector<unsigned short int> depths(num_points);
        std::vector<unsigned short int> types(num_points);
        // -- reserve set size
        local_cache_cell_index.reserve(INIT_CELL_CACHE_SIZE);
        local_cache_types.reserve(INIT_CELL_CACHE_SIZE);
        local_cache_depths.reserve(INIT_CELL_CACHE_SIZE);

        // Determine needed cells
        #pragma omp parallel for default(none) shared(num_points, input_mapping, points, depths, types)
        for (size_t i = 0; i < num_points; ++i) {
            double point[num_dim];
            for (size_t j = 0; j < num_dim; ++j) {
                point[j] = points[j][i];
            }
            input_mapping.at(i) = compute_cell_mapping(point, depths.at(i),  types.at(i));
        }
        // Determine unique cells needed
        size_t num_cells = 0;
        for (size_t j = 0; j < num_points; j++){
            bool found = false;
            for (size_t i = 0; i < num_cells; ++i) {
                if (input_mapping.at(j) == local_cache_cell_index[i]){
                    input_mapping.at(j) = i;
                    found = true;
                    break;
                }
            }
            if (!found){
                local_cache_cell_index.push_back(input_mapping.at(j));
                local_cache_depths.push_back(depths.at(j));
                local_cache_types.push_back(types.at(j));
                input_mapping.at(j) = num_cells;
                num_cells++;
            }
        }

        // finish cache setup
        std::vector<double> local_cache(num_cells*weight_size);

        // load needed cells
        #pragma omp parallel for default(none) shared(num_cells, local_cache_cell_index, local_cache_depths, local_cache, node_values, weight_offset, domain ,dx)
        for (size_t i = 0; i < num_cells; ++i) {
            size_t local_cell_index[num_dim];
            un_global_indexing(local_cache_cell_index.at(i), local_cell_index);
            auto depth_diff = max_depth - local_cache_depths.at(i);
            for (size_t j = 0; j < (1<<num_dim); ++j) {
                size_t point_index = compute_global_index_of_corner(local_cell_index, j, depth_diff);
                local_cache.at(i*weight_size+j) = node_values[mphf(point_index)];
            }
            // Load domain info into weights
            auto cell_edge_index_size = 1 << depth_diff;
            for (size_t j = 0; j != num_dim; ++j) {
                local_cache.at(i*weight_size+weight_offset+j) = domain[j*2]+dx[j]*local_cell_index[j];
                local_cache.at(i*weight_size+weight_offset+num_dim+j) = domain[j*2]+dx[j]*(local_cell_index[j] + cell_edge_index_size);
            }
        }

        // Do interpolation on cells
        #pragma omp parallel for simd default(none) shared(return_array, points, num_points, local_cache, input_mapping)
        for (size_t i = 0; i < num_points; ++i) {
            double point[num_dim];
            for (size_t j = 0; j < num_dim; ++j) {
                point[j] = points[j][i];
            }
            return_array[i] = nd_linear_interp(point, &(local_cache.at(weight_size*input_mapping.at(i))));//interp_point(point, &(local_cache.at(weight_size*input_mapping.at(i))));
        }

//        // distribute points to interpolate among threads
//        #pragma omp parallel default(shared)    // NOLINT(openmp-use-default-none)
//        {
//            double local_weights[weight_size];
//            bool first_iteration = true;
//            std::cout << "Thread: " << omp_get_thread_num() << std::endl;
//            #pragma omp for schedule(auto)
//            for (unsigned int i = 0; i < num_points; i++){
//                // load points into local array
//                double point_domain[num_dim];
//                double point[num_dim];
//                for (size_t j = 0; j != num_dim; j++){
//                    point_domain[j] = points[j][i];
//                    point[j] = point_domain[j];
//                    // Do any needed domain transforms
//                    domain_transform(&point_domain[j], j, 1);
//                }
//                // Check if current loaded cell is the correct cell
//                if (first_iteration || !point_in_current_cell(point_domain, local_weights)){
//                    // load cell
//                    first_iteration = false;
//                    load_cell(point_domain, local_weights);
//                }
//                // do interpolation
//                return_array[i] = interp_point(point, local_weights);
//            }
//        }
    }

private:
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
    // data for point array
    indexing_int point_cache_indices[POINT_CACHE_SIZE];
    double point_cache_values[POINT_CACHE_SIZE];
    size_t point_cache_start;
    size_t point_cache_end;
    std::mutex point_cache_edit;
    std::shared_mutex point_cache_range;
    // data for cell array
    indexing_int cell_cache_indices[CELL_CACHE_SIZE];
    double cell_cache_values[CELL_CACHE_SIZE*((1<<num_dim)+num_dim*2)];
    size_t cell_cache_start;
    size_t cell_cache_end;
    std::mutex cell_cache_edit;
    std::shared_mutex cell_cache_range;

    void compute_cartesian_indices(const double *point, size_t *cartesian_indices) const {
        for (size_t i = 0; i != num_dim; i++){
            // restrict the point to fall within the real domain (as opposed to the index domain)
            double p = std::max(std::min(point[i], domain[i * 2 + 1]), domain[i * 2 + 0]);
            // compute cartesian index
            cartesian_indices[i] = size_t((p - domain[i * 2 + 0]) / dx[i]);
            // If the index is outside the index domain of the emulator round to the nearest cell.
            cartesian_indices[i] = std::max(size_t(0), cartesian_indices[i]);
            cartesian_indices[i] = std::min(max_index, cartesian_indices[i]);
        }
    }

    void update_shared_cell_array(unsigned short int type, size_t cell_index, const double* cell_weights){
        // Update shared cell cache with local array
        cell_cache_edit.lock();    // Lock ability to edit array sections not covered by [start, end)
        // Make sure it was not added since you last checked and before you got the lock
        for (size_t i = cell_cache_start; i != cell_cache_end; i=(i+1)%CELL_CACHE_SIZE) {
            if (cell_index == cell_cache_indices[i]){
                cell_cache_edit.unlock();
                return;
            }
        }
        // load vars into mutable section of tables. Note that A[end%N] is always available to be mutated, as start!=end
        cell_cache_indices[cell_cache_end] = cell_index;
        for (size_t i = 0; i < weight_size; ++i) {
            cell_cache_values[cell_cache_end*weight_size + i] = cell_weights[i];
        }
        // precompute the new indices
        auto new_end = (cell_cache_end + 1) % CELL_CACHE_SIZE;
        auto new_start = cell_cache_start;
        if (new_start == new_end){
            new_start = (new_start + 1) % CELL_CACHE_SIZE;
        }
        // update indices
        cell_cache_range.lock();    // Lock ranges while the editing thread updates it
        cell_cache_end = new_end;
        cell_cache_start = new_start;
        cell_cache_range.unlock();  // release range lock
        cell_cache_edit.unlock();    // release edit lock
    }

    bool try_local_cell_load(const size_t& cell_index, double* local_weights){
        // determine if this cell is loaded. If it is, copy weights into local array
        // Load cell from local shared array if it is there, if not, load from global array and add it to shared array
        // TODO: switch arrays based on type
        // Check if cell is already loaded
        cell_cache_range.lock_shared();
        for (size_t i = cell_cache_start; i != cell_cache_end; i=(i+1)%CELL_CACHE_SIZE) {
            if (cell_index == cell_cache_indices[i]){
                for (size_t j = 0; j < weight_size; ++j) {
                    local_weights[j] = cell_cache_values[i*weight_size+j];
                }
                cell_cache_range.unlock_shared();
                return true;
            }
        }
        cell_cache_range.unlock_shared();
        return false;
    }

    void load_point(const unsigned short int type, const size_t point_index, double& cell_weight){
        // Load point from local shared array if it is there, if not, load from global array and add it to shared array
        // TODO: switch arrays based on type
        // Check if point is already loaded
        point_cache_range.lock_shared();
        for (size_t i = point_cache_start; i != point_cache_end; i=(i+1)%POINT_CACHE_SIZE) {
            if (point_index == point_cache_indices[i]){
                cell_weight = point_cache_values[i];
                point_cache_range.unlock_shared();
                return;
            }
        }
        point_cache_range.unlock_shared();
        // If not found, load the element from global memory
        cell_weight = node_values[mphf(point_index)];
        // update shared point array
        point_cache_edit.lock();    // Lock ability to edit array sections not covered by [start, end)
        // Make sure it was not added since you last checked and before you got the lock
        for (size_t i = point_cache_start; i != point_cache_end; i=(i+1)%POINT_CACHE_SIZE) {
            if (point_index == point_cache_indices[i]){
                point_cache_edit.unlock();
                return;
            }
        }
        // load vars into mutable section of tables. Note that A[end%N] is always available to be mutated, as start!=end
        point_cache_values[point_cache_end] = cell_weight;
        point_cache_indices[point_cache_end] = point_index;
        // precompute the new indices
        auto new_end = (point_cache_end + 1) % POINT_CACHE_SIZE;
        auto new_start = point_cache_start;
        if (new_start == new_end){
            new_start = (new_start + 1) % POINT_CACHE_SIZE;
        }
        // update indices
        point_cache_range.lock();    // Lock ranges while the editing thread updates it
        point_cache_end = new_end;
        point_cache_start = new_start;
        point_cache_range.unlock();  // release range lock
        point_cache_edit.unlock();    // release edit lock
    }

    inline size_t compute_global_index(const size_t* cell_indices){
        size_t global_index = 0;
        for (unsigned int i = 0; i != num_dim; ++i) {
            global_index += cell_indices[i]*(cell_index_transform_weights[i]);
        }
        return global_index;
    }

    inline size_t trimmed_and_compute_global_index(size_t* cell_indices, const unsigned short int depth_diff){
        size_t global_index = 0;
        for (unsigned int i = 0; i != num_dim; ++i) {
            cell_indices[i] = ((cell_indices[i]>>depth_diff)<<depth_diff);
            global_index += cell_indices[i]*(cell_index_transform_weights[i]);
        }
        return global_index;
    }

    inline size_t compute_global_index_of_corner(const size_t* cell_indices, unsigned int corner,
                                                 unsigned short int depth_diff){
        size_t cell_edge_size = (1<<depth_diff);
        size_t global_index = 0;
        for (unsigned int i = 0; i != num_dim; ++i) {
            if ((corner >> i) & 1){
                global_index += (cell_indices[i] + cell_edge_size)*(point_index_transform_weights[i]);
            } else{
                global_index += cell_indices[i]*(point_index_transform_weights[i]);
            }
        }
        return global_index;
    }

    void load_cell(double* input, double* cell_weights){
        /*
         * Determines the cell needed for a given input and loads the cell weights into the cell_weights vector.
         * Updates the shared cell weight array.
         * Can be called in parallel.
         */
        // determine needed cells
        unsigned short int depth;
        unsigned short int type;
        size_t local_cell_index[num_dim];
        auto cell_index = compute_cell_mapping(input, depth, type, local_cell_index);
        auto depth_diff = max_depth - depth;
        // Check if cell is loaded into shared local array and load it if it is.
        if (!try_local_cell_load(cell_index, cell_weights)){
            // determine points needed and load them into the weigh vector
            const size_t num_points = (1<<num_dim);
            for (unsigned int i = 0; i < num_points; ++i) { // i is also the location in the weight vector
                // determine global index of point
                size_t point_index = compute_global_index_of_corner(local_cell_index, i, depth_diff);
                // Load point
                load_point(type, point_index,cell_weights[i]);
            }
            // load domain information into weight vector
            // lower and upper corner
            auto cell_edge_index_size = 1 << depth_diff;
            for (size_t i = 0; i != num_dim; ++i) {
                cell_weights[weight_offset+i] = domain[i*2]+dx[i]*local_cell_index[i];
                cell_weights[weight_offset+num_dim+i] = domain[i*2]+dx[i]*(local_cell_index[i] + cell_edge_index_size);
            }
            // update shared cell array with weights
            update_shared_cell_array(type, cell_index, cell_weights);
        }

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

    inline void un_global_indexing(size_t global_index, size_t local_index[num_dim]){
        for (unsigned int i = 0; i < num_dim; ++i) {
            local_index[i] = global_index / cell_index_transform_weights[i];
            global_index = global_index % cell_index_transform_weights[i];
        }
    }

    // compute the depth, type, global_index and cell_index for the given input_point
    size_t compute_cell_mapping(double* input_point, unsigned short int& depth, unsigned short int& type, size_t* cell_index){
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
    size_t compute_cell_mapping(double* input_point, unsigned short int& depth, unsigned short int& type){
        size_t cell_index[num_dim];
        return compute_cell_mapping(input_point, depth, type, cell_index);
    }

    bool point_in_current_cell(const double* point_domain, const double* weights){
        // determine if point is in current domain.
        const double* cell_domain = get_cell_domain(weights);
        for (size_t i = 0; i != num_dim; i++){
            if ((point_domain[i] < cell_domain[i])
            || (point_domain[i] > cell_domain[i+num_dim])){
                return false;
            }
        }
        return true;
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
        for (size_t i = 0; i != num_dim; i++){
            x[i] = (point[i] - cell_domain[i]) / (cell_domain[num_dim + i] - cell_domain[i]);
        }
        // apply weights
        double solution = 0;
        for (size_t i = 0; i != weight_offset; i++){
            double w = 1;
            for (size_t j = 0; j != num_dim; j++){
                auto bit = (i >> j) & 1;
                w *= bit == 0 ? (1 - x[j]) : x[j];
            }
            solution += weight[i] * w;
        }
        return solution;
    }

};

#endif //CPP_EMULATOR_EMULATOR_H
