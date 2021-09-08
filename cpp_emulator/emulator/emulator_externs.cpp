//
// Created by jared on 5/31/2021.
//
#include "emulator.h"
#include "table_params.h"

extern "C" {
    /**
     * Creates an instance of the emulator based on the parameterized in "table_params.h" and changes the emulator
     * pointer to point to it.
     *
     * @param filename: location of the hdf5 file containing the table
     * @param mapping_filename: location of the bin file that contains the pthash mapping function
     * @param emulator: a void pointer to be changed to point to the newly instantiated emulator
     */
    void ND_TREE_EMULATOR_NAME_SETUP(const char* filename, const char* mapping_filename, void*& emulator) {
       emulator = (void*)(new Emulator<ND_TREE_EMULATOR_TYPE>(filename, mapping_filename));
    }

    /**
     * Given a pointer to the emulator, takes N arrays of size num_points, each entry corresponding to a different
     * point, and calls the emulator to do the interpolation. The results are returned through the return array.
     *
     * Utilizes OpenMP parallelism in a thread safe way.
     *
     * @param emulator: pointer to emulator created by setup function
     * @param POINT_INPUTS: expands to D different double* xi parameters, for i in D, where D is the number of dimensions
     * of the table.
     * @param num_points: Number of points in each array
     * @param return_array: where the interpolated values for each point are stored.
     */
    void ND_TREE_EMULATOR_NAME_INTERPOLATE(void*& emulator, POINT_INPUTS, size_t& num_points, double* return_array) {
       POINT_GROUPING;
       ((Emulator<ND_TREE_EMULATOR_TYPE> *)emulator)->interpolate(points, num_points, return_array);
    }

    /**
     * Similar to ND_TREE_EMULATOR_NAME_INTERPOLATE except each expended values in POINT_INPUTS is a double. Thus you
     * pass in a single point at a time.
     *
     * This uses a single cell caching system per a thread. It is thread safe.
     */
    void ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE(void*& emulator, POINT_INPUTS_SINGLE, double& return_array) {
       POINT_GROUPING_SINGLE;
       ((Emulator<ND_TREE_EMULATOR_TYPE> *)emulator)->interpolate(points, return_array);
    }

    /**
     * Similar to ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE except it also returns the derivative along the second dimension.
     * Adding other dimension should be straight forward. It is done compile time so a new function would need to be made
     * for each dimension.
     */
    void ND_TREE_EMULATOR_NAME_INTERPOLATE_SINGLE_DX1(void*& emulator, POINT_INPUTS_SINGLE, double& return_array, double& dy_dx1) {
       POINT_GROUPING_SINGLE;
       ((Emulator<ND_TREE_EMULATOR_TYPE> *)emulator)->interpolate<1>(points, return_array, dy_dx1);
    }

    /**
     * Deallocate the emulator.
     *
     * @param: pointer to the emulator to deallocate.
     */
    void ND_TREE_EMULATOR_NAME_FREE(void*& emulator){
       delete((Emulator<ND_TREE_EMULATOR_TYPE>*)emulator);
    }
}