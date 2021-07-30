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

#define N 100
#define N_vars 400
#define M 10000
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
        // get a random element from the array. If it is not there, add it
        int pulled_var = 0;
        // check if it is in the array already
        for (unsigned int i = start; i != end; i=(i + 1) % N) {
            if (var == array[i]){
                return local_array[i];
            }
        }
        // If it was not found, add the element
        array[end] = var;
        local_array[end] = global_array[var];
        auto return_vale = local_array[end];
        end = (end + 1) % N;
        if (start == end){
            start = (start + 1) % N;
        }
        return return_vale;
    }
    string get_array(){
        stringstream str_out;
        for (int i = 0; i < N; ++i) {
            str_out << std::setfill (' ') << std::setw (2) << array[i] << " ";
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