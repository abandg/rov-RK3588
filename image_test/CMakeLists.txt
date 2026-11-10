cmake_minimum_required(VERSION 3.11)
project(rk3588_image_test LANGUAGES CXX)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc)
find_library(RKNN_RUNTIME_LIBRARY NAMES rknnrt HINTS "${CMAKE_CURRENT_SOURCE_DIR}/runtime" PATHS /lib)
if(NOT RKNN_RUNTIME_LIBRARY)
    message(FATAL_ERROR "Could not find librknnrt.so. Set RKNN_RUNTIME_LIBRARY to its full path.")
endif()

add_executable(image_probe image_probe.cpp)
target_compile_features(image_probe PRIVATE cxx_std_14)
target_include_directories(image_probe PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_probe PRIVATE ${OpenCV_LIBS} ${RKNN_RUNTIME_LIBRARY})
