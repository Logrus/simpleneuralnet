
project(simpleneuralnet_tests)
 
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

# Define tests targets
add_executable(sample_unittest sample_unittest.cpp)
target_link_libraries(sample_unittest gtest_main gtest)


add_test(NAME simpleneuralnet_tests COMMAND sample_unittest)

