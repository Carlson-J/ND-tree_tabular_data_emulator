cmake_minimum_required(VERSION 3.16)
project(tests)

set(CMAKE_CXX_STANDARD 17)

add_executable(tests_all test_setup.cpp)

target_link_libraries(tests_all HighFive morton-nd::MortonND)