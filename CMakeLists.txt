cmake_minimum_required(VERSION 3.11)
project(rk3588_vision LANGUAGES CXX)

set(RK3588_REALTIME_TARGET rk3588_vision)
add_subdirectory(realtime_test)
set_target_properties(rk3588_vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
