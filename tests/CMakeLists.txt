
project(simpleneuralnet_tests)
 
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

set(AffineLayer_CPP ../affinelayer.cpp
                    ../layer.cpp
                    ../gradientcheck.h
                    )

# Define tests targets
add_executable(affinelayer_unittest affinelayer_unittest.cpp ${AffineLayer_CPP})
target_link_libraries(affinelayer_unittest gtest_main gtest)


add_test(NAME simpleneuralnet_tests COMMAND affinelayer_unittest)

