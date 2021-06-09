#include <iostream>

extern "C" {
    void test_(double* array, int* size_x, int* size_y);
}


int main(){
    std::cout << "Starting test" << std::endl;

    // build array
    int size_x = 4; // fastest varying in memory
    int size_y = 3;
    double array[size_x*size_y];
    for (size_t i = 0; i < size_y; i++){
        for (size_t j = 0; j < size_x; j++){
            array[j + i*size_x] = 0;
        }
    }
    // send array to fortran func
    test_(array, &size_x, &size_y);

    // print array
    for (size_t i = 0; i < size_y; i++){
        for (size_t j = 0; j < size_x; j++){
            std::cout << array[j + i*size_x] << " ";
        }
        std::cout << std::endl;
    }
}