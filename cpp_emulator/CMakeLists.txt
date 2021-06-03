cmake_minimum_required(VERSION 3.16)
project(profiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -Wall -Wextra")

add_subdirectory(HighFive)
include_directories(emulator)

add_executable(profiler profiler/main.cpp)
add_compile_options(-DHIGHFIVE_USE_BOOST=OFF)
target_link_libraries(profiler HighFive)

add_subdirectory(Tests)

# Create library for emulator
project(ND_emulator_lib DESCRIPTION "library for using the ND tree emulator")
add_library(ND_emulator_lib SHARED ./emulator/emulator_externs.cpp ./emulator/emulator.h)
target_link_libraries(ND_emulator_lib HighFive)

set_target_properties(ND_emulator_lib PROPERTIES LINKER_LANGUAGE CXX)
#set_target_properties(ND_emulator_lib PROPERTIES PUBLIC_HEADER ./emulator.h)

#include(GNUInstallDirs)
#
#install(TARGETS ND_emulator_lib
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})