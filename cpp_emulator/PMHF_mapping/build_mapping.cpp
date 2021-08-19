//
// Created by carls502 on 7/19/2021.
//
#include <iostream>
#include "../HighFive/include/highfive/H5File.hpp"
#include "../pthash/include/pthash.hpp"
#include "../pthash/src/util.hpp"  // for functions distinct_keys and check
#include "string"
#include <cstdlib>

int main(int argc, char* argv[]) {
    using namespace pthash;
    // check that an input was given
    if (argc != 4){
        std::cerr << "Both the location of the hdf5 file for the table and the output name must be given. Only " << argc - 1 << " arguments where given" << std::endl;
        return -1;
    }
    std::string file_location = argv[1];
    std::string output_filename = argv[2];
    size_t num_points = atoll(argv[3]);
    std::vector<unsigned long> indexing(num_points);
    std::vector<double> node_values(num_points);
    std::vector<double> new_node_values(num_points);
    std::vector<double> new_node_values_testing(num_points);
    // Load mapping data
    HighFive::File file(file_location, HighFive::File::ReadWrite);
    HighFive::Group mapping_group = file.getGroup("mapping");
    HighFive::DataSet dataset = mapping_group.getDataSet("indexing");
    dataset.template read(indexing);
    dataset = mapping_group.getDataSet("node_values");
    dataset.template read(node_values);

    /* Set up a build configuration. */
    build_configuration config;
    config.c = 2.5;//6.0;
    config.alpha = 0.99;
    config.minimal_output = true;  // mphf
    config.verbose_output = true;
//    config.num_partitions = 3;
    /* Declare the PTHash function. */
    typedef single_phf<murmurhash2_64,         // base hasher
            compact_compact,  // encoder type
            true                    // minimal
    >
            pthash_type;

    // config.num_partitions = 50;
    // config.num_threads = 4;
    // typedef partitioned_mphf<murmurhash2_64,        // base hasher
    //                          dictionary_dictionary  // encoder type
    //                          >
    //     pthash_type;

    pthash_type f;

    /* Build the function in internal memory. */
    std::cout << "building the function..." << std::endl;
    auto start = clock_type::now();
    auto timings = f.build_in_internal_memory(indexing.begin(), indexing.size(), config);
    // auto timings = f.build_in_external_memory(keys.begin(), keys.size(), config);
    double total_seconds = timings.partitioning_seconds + timings.mapping_ordering_seconds +
                           timings.searching_seconds + timings.encoding_seconds;
    std::cout << "function built in " << seconds(clock_type::now() - start) << " seconds"
              << std::endl;
    std::cout << "computed: " << total_seconds << " seconds" << std::endl;

    /* Compute and print the number of bits spent per key. */
    double bits_per_key = static_cast<double>(f.num_bits()) / f.num_keys();
    std::cout << "function uses " << bits_per_key << " [bits/key]" << std::endl;

    /* Sanity check! */
    if (check(indexing.begin(), f)) std::cout << "EVERYTHING OK!" << std::endl;

    /* Now rearrange data to be in the correct place */
    for (uint64_t i = 0; i != num_points; ++i) {
        auto new_index = f(indexing[i]);
//        std::cout << "f(" << (int)indexing[i] << ") = " << f(indexing[i]) << ", storing " << node_values[i] << '\n';
        new_node_values[new_index] = node_values[i];
    }

    /* Serialize the data structure to a file. */
    std::cout << "serializing the function to disk..." << std::endl;
    essentials::save(f, output_filename.c_str());

    {
        /* Now reload from disk and query. */
        pthash_type other;
        essentials::load(other, output_filename.c_str());
        for (uint64_t i = 0; i != 10; ++i) {
            std::cout << "f(" << static_cast<int>(indexing[i]) << ") = " << other(indexing[i]) << '\n';
            assert(f(indexing[i]) == other(indexing[i]));
        }
    }

    // Save in hdf5 table file
    mapping_group.createDataSet<double>("node_values_encoded", HighFive::DataSpace::From(new_node_values));
    dataset = mapping_group.getDataSet("node_values_encoded");
    dataset.write(new_node_values);

    // Test to make sure it worked.
    dataset.template read(new_node_values_testing);
    for (unsigned int i = 0; i != num_points; ++i) {
        if (new_node_values[i] != new_node_values_testing[i]){
            std::cout << "Error saving values!" << std::endl;
            return -1;
        }
        if (node_values[i] != new_node_values_testing[f(indexing[i])]){
            std::cout << "i:" << i << "->" << indexing[i] << ", n1:" << node_values[i] << ", n2:" << new_node_values_testing[f(indexing[i])]
             << " = " << "N[" << f(indexing[i]) << "]" << std::endl;
            std::cout << "Error mapping values!" << std::endl;
            return -1;
        }
    }
    std::cout << "Test Passed!" << std::endl;

    return 0;
}