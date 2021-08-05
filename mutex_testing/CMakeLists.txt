cmake_minimum_required(VERSION 3.20)
project(mutex_testing)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(mutex_testing main.cpp)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fopenmp")
add_executable(test_design test_design.cpp)