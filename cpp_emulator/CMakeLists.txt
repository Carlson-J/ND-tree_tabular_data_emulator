cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -Wall -Wextra -fopenmp -march=native -std=c++17")

project(build_mapping)
add_executable(make_mapping PMHF_mapping/build_mapping.cpp)
target_link_libraries(make_mapping HighFive)

project(profiler)
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
# add_subdirectory(pthash)
add_library(ND_emulator_lib SHARED ./emulator/emulator_externs.cpp ./emulator/emulator.h)
#target_link_libraries(ND_emulator_lib HighFive)
target_link_libraries(ND_emulator_lib PRIVATE morton-nd::MortonND HighFive)

if (DEFINED ENV{EMULATOR_NAME})
SET(EMULATOR_NAME $ENV{EMULATOR_NAME})
else (DEFINED ENV{EMULATOR_NAME})
SET(EMULATOR_NAME "emulator")
endif (DEFINED ENV{EMULATOR_NAME})

set_target_properties(ND_emulator_lib PROPERTIES PREFIX "")
set_target_properties(ND_emulator_lib PROPERTIES OUTPUT_NAME "${EMULATOR_NAME}")
set_target_properties(ND_emulator_lib PROPERTIES LINKER_LANGUAGE CXX)
#set_target_properties(ND_emulator_lib PROPERTIES PUBLIC_HEADER ./emulator.h)
#
#include(GNUInstallDirs)
#
#install(TARGETS ND_emulator_lib
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})