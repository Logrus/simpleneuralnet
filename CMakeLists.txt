cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

option(test "Build all tests." ON)

add_definitions(-std=c++11)

find_package(Eigen3)
include_directories(${Eigen3_INCLUDES})

set(Project_CPP
    main.cpp
    solver.cpp
    net.cpp
    layer.cpp
    affinelayer.cpp
    )
set(Project_H
    solver.h
    net.h
    layer.h
    affinelayer.h
    gradientcheck.h
    )

# Project internals
add_executable(main ${Project_CPP})


if(test)
  add_subdirectory(tests)
endif()

