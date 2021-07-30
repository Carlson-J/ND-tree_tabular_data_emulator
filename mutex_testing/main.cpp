#include <iostream>
#include <thread>         // std::thread
#include <mutex>          // std::mutex
#include <omp.h>

using namespace std;
#define NUM_THREADS 4

int main() {
    // set number of threads and disable dynamic thread number option
    omp_set_num_threads(NUM_THREADS);
    omp_set_dynamic(0);

    cout << endl << "No mutex:" << endl;
    #pragma omp parallel for default(none) shared(std::cout)
    for (int i = 0; i < 10; ++i) {
        auto id = omp_get_thread_num ();
        std::cout << "ID: " << id <<  " Hello, World!" << std::endl;
    }

    cout << endl << "Using mutex:" << endl;
    mutex mtx;
    #pragma omp parallel for default(none) shared(std::cout, mtx)
    for (int i = 0; i < 10; ++i) {
        auto id = omp_get_thread_num ();
        mtx.lock();
        std::cout << "ID: " << id <<  " Hello, World!" << std::endl;
        mtx.unlock();
    }
    return 0;
}
