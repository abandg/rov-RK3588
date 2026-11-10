cmake_minimum_required(VERSION 3.11)
project(rk3588_vision LANGUAGES CXX)

find_package(OpenCV 4 REQUIRED COMPONENTS core highgui videoio)

add_executable(rk3588_vision src/main.cpp)
target_compile_features(rk3588_vision PRIVATE cxx_std_14)
target_include_directories(rk3588_vision PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rk3588_vision PRIVATE ${OpenCV_LIBS})
