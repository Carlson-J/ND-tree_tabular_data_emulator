cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
if (CXX EQUAL "icpc")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -fopenmp -march=native -mtune=native -g -Wall -Wextra -no-inline-max-size -no-inline-max-total-size")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -march=native -fopenmp -g -Wall -Wextra ")
endif()
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -fopenmp -march=native -mtune=native -g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -march=native -fopenmp -g -Wall -Wextra ")

project(build_mapping)
add_executable(make_mapping PMHF_mapping/build_mapping.cpp)
target_link_libraries(make_mapping HighFive)

project(profiler)
add_compile_options(-DHIGHFIVE_USE_BOOST=OFF)

add_subdirectory(HighFive)
include_directories(emulator)

add_executable(profiler profiler/main.cpp)


target_link_libraries(profiler PRIVATE HighFive)

add_subdirectory(Tests)

# Create library for emulator
project(ND_emulator_lib DESCRIPTION "library for using the ND tree emulator")
# add_subdirectory(pthash)
add_library(ND_emulator_lib SHARED ./emulator/emulator_externs.cpp ./emulator/emulator.h)
target_link_libraries(ND_emulator_lib PRIVATE HighFive)

if (DEFINED ENV{EMULATOR_NAME})
SET(EMULATOR_NAME $ENV{EMULATOR_NAME})
else (DEFINED ENV{EMULATOR_NAME})
SET(EMULATOR_NAME "emulator")
endif (DEFINED ENV{EMULATOR_NAME})

set_target_properties(ND_emulator_lib PROPERTIES PREFIX "")
set_target_properties(ND_emulator_lib PROPERTIES OUTPUT_NAME "${EMULATOR_NAME}")
set_target_properties(ND_emulator_lib PROPERTIES LINKER_LANGUAGE CXX)
