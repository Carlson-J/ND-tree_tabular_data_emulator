cmake_minimum_required(VERSION 3.16)
project(cpp_emulator)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(HighFive)

add_executable(cpp_emulator main.cpp)
add_compile_options(-DHIGHFIVE_USE_BOOST=ON)
target_link_libraries(cpp_emulator HighFive)

add_subdirectory(Tests)