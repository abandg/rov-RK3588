cmake_minimum_required(VERSION 3.11)
project(rk3588_realtime_test LANGUAGES CXX)

find_package(OpenCV 4 REQUIRED COMPONENTS core highgui imgproc videoio)
find_library(RKNN_RUNTIME_LIBRARY NAMES rknnrt HINTS "${CMAKE_CURRENT_SOURCE_DIR}/runtime" PATHS /lib)
if(NOT RKNN_RUNTIME_LIBRARY)
    message(FATAL_ERROR "Could not find librknnrt.so. Set RKNN_RUNTIME_LIBRARY to its full path.")
endif()

if(NOT DEFINED RK3588_REALTIME_TARGET)
    set(RK3588_REALTIME_TARGET realtime_test)
endif()

add_executable(${RK3588_REALTIME_TARGET} main.cpp)
target_compile_features(${RK3588_REALTIME_TARGET} PRIVATE cxx_std_14)
target_include_directories(${RK3588_REALTIME_TARGET} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${RK3588_REALTIME_TARGET} PRIVATE ${OpenCV_LIBS} ${RKNN_RUNTIME_LIBRARY})
