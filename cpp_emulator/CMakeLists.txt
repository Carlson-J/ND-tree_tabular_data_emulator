cmake_minimum_required(VERSION 3.20)
project(profiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -Wall -Wextra -fopenmp")
add_compile_options(-DHIGHFIVE_USE_BOOST=OFF)

add_subdirectory(HighFive)
include_directories(emulator)

add_subdirectory(./morton-nd)
#find_package(morton-nd CONFIG REQUIRED)

add_executable(profiler profiler/main.cpp)

target_link_libraries(profiler PRIVATE morton-nd::MortonND HighFive)
#target_link_libraries(profiler HighFive)

add_subdirectory(Tests)

# Create library for emulator
project(ND_emulator_lib DESCRIPTION "library for using the ND tree emulator")
add_library(ND_emulator_lib SHARED ./emulator/emulator_externs.cpp ./emulator/emulator.h)
#target_link_libraries(ND_emulator_lib HighFive)
target_link_libraries(ND_emulator_lib PRIVATE morton-nd::MortonND HighFive)

set_target_properties(ND_emulator_lib PROPERTIES LINKER_LANGUAGE CXX)
#set_target_properties(ND_emulator_lib PROPERTIES PUBLIC_HEADER ./emulator.h)

#include(GNUInstallDirs)
#
#install(TARGETS ND_emulator_lib
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})