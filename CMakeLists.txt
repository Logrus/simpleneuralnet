cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

option(test "Build all tests." ON)

if(test)
  add_subdirectory(tests)
endif()

