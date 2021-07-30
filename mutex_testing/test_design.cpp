//
// Created by carls502 on 7/30/2021.
//
#include <iostream>
#include <thread>         // std::thread
#include <mutex>          // std::mutex
#include <shared_mutex>
#include <omp.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>     // std::cout, std::endl
#include <iomanip>      // std::setfill, std::setw
#include "ctime"

#define N 10
#define N_vars 40
#define M 10000000
#define NUM_THREADS 4
using namespace std;

class Loader{
public:
    Loader() {
        for (auto& i : array) {
            i = -1;
        }
        for (int i = 0; i < N_vars; ++i) {
            global_array[i] = (float)i*2.0f;
        }
    };
    float load_element(int var){
        /*
         * Loading an element is done from a local array. The element id is search for in an id array. If it is found,
         * the data from the local array that the id array points to is loaded and returned. This can be done as long
         * as the start and stop indices are not being updated, i.e., each reader has a shared mutex.
         * The start and stop indices show which part of the cyclical array are currently loaded into the local array.
         * It will start at 0 elements and then grow to N-1 in size. Once to the max size, both start and end increase
         * by one whenever a new element is added. This allows for one spot on the array that is free to be changed
         * without stopping the reads.
         * An editor mutex is used to keep only one thread from editing the editable portion of the array. Once it is
         * done editing, it takes a unique mutex that stops all other threads from accessing the local array. It then
         * updates the indices and returns both of its locks.
         */
        // get an element from the array. If it is not there, add it
        // check if it is in the array already
        update_range_mtx.lock_shared();
        for (unsigned int i = start; i != end; i=(i + 1) % N) {
            if (var == array[i]){
                auto return_var = local_array[i];
                update_range_mtx.unlock_shared();
                return return_var;
            }
        }
        update_range_mtx.unlock_shared();
        // If it was not found, add the element
        edit_array_mtx.lock();  // Lock ability to edit array sections not covered by [start, end)
        // load vars into mutable section of tables. Note that A[end%N] is always available to be mutated, as start!=end
        array[end] = var;
        local_array[end] = global_array[var];
        // Update local cache/vec/scalar
        auto return_vale = local_array[end];
        // precompute the new indices
        auto new_end = (end + 1) % N;
        auto new_start = start;
        if (new_start == new_end){
            new_start = (new_start + 1) % N;
        }
        // update indices
        update_range_mtx.lock();    // Lock ranges while the editing thread updates it
        end = new_end;
        start = new_start;
        update_range_mtx.unlock();  // release range lock
        edit_array_mtx.unlock();    // release edit lock
        return return_vale;
    }
    string get_array(){
        stringstream str_out;
        for (int i : array) {
            str_out << std::setfill (' ') << std::setw (2) << i << " ";
        }
        return str_out.str();
    }
    int array[N];
    float local_array[N];
    float global_array[N_vars];
private:
    unsigned int start{0};
    unsigned int end{0};
    mutex edit_array_mtx;
    shared_mutex update_range_mtx;

};


int main(){
    // set number of threads and disable dynamic thread number option
    omp_set_num_threads(NUM_THREADS);
    omp_set_dynamic(0);

    mutex print_mtx;
    // create loader
    Loader loader;
    // print out initial array
    cout << "Initial array" << endl << loader.get_array() << endl;
    // add elements and print out array each time
    #pragma omp parallel default(none) shared(loader, cout, print_mtx)
    {
        unsigned int id = omp_get_thread_num();
        srand(id*time(0));
        #pragma omp for
        for (int i = 0; i < M; ++i) {
            // Create random number between 0 and N
            int var = rand() % N_vars;
            float loaded_var = loader.load_element(var);
            if (((float)var*2.0f) != loaded_var){
                print_mtx.lock();
                cout << "id:" << std::setfill (' ') << std::setw (2) << id << " Failed to get right element (" << loaded_var << "!=" << ((float)var*2.0f)  << ")" << endl;
                print_mtx.unlock();
            }
//            print_mtx.lock();
//            cout  << "id:" << std::setfill (' ') << std::setw (2) << id << " " << loader.get_array() << endl;
//            print_mtx.unlock();
        }
    }



}