cmake_minimum_required(VERSION 3.5)
project(interp LANGUAGES CXX VERSION 3.8.5)

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Debug")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(interp interp.cc)
