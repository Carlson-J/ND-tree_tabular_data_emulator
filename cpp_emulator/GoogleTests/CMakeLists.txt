# 'Google_test' is the subproject name
project(Google_tests)

# 'lib' is the folder with Google Test sources
add_subdirectory(lib)
include_directories(googletest/include googletest)

# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(Google_Tests_run test_initialization.cpp)
target_link_libraries(Google_Tests_run gtest gtest_main)

target_include_directories(
        Google_Tests_run
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../emulator
)