cmake_minimum_required(VERSION 3.16)
project(cpp_emulator)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(GoogleTests)

add_executable(cpp_emulator emulator/main.cpp emulator/emulator.h GoogleTests)

