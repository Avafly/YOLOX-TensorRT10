cmake_minimum_required(VERSION 3.20)
project(YOLOX-TRT10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# compile options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(LIB_ROOT "$ENV{HOME}/Documents/libs")

# opencv
set(OpenCV_DIR "${LIB_ROOT}/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(Iconv QUIET)
# cuda
set(CUDAToolkit_ROOT "/usr/local/cuda-12.9")
find_package(CUDAToolkit REQUIRED)

# detect_single
add_executable(detect_single
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_single.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detector/yolox_detector.cpp
)
target_include_directories(detect_single PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/detector
)
target_link_libraries(detect_single
    ${OpenCV_LIBS}
    nvonnxparser
    nvinfer
    CUDA::cudart
)

# detect_batch
add_executable(detect_batch
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detector/yolox_detector.cpp
)
target_include_directories(detect_batch PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/detector
)
target_link_libraries(detect_batch
    ${OpenCV_LIBS}
    nvonnxparser
    nvinfer
    CUDA::cudart
)
