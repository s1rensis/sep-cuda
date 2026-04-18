cmake_minimum_required(VERSION 3.20)
project(sepcuda-minimal LANGUAGES CXX)

find_package(SEPCUDA CONFIG REQUIRED)

add_executable(sepcuda_minimal main.cpp)
target_link_libraries(sepcuda_minimal PRIVATE SEPCUDA::SEPCUDA)
