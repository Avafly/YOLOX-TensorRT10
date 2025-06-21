cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0146 OLD)   # for disabling FindCUDA warning
project(YOLOX-TRT10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")

set(LIB_ROOT "$ENV{HOME}/Documents/libs")

# opencv
set(OpenCV_DIR "${LIB_ROOT}/opencv_4.12.0_cuda/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(Iconv QUIET)
# cuda
find_package(CUDA REQUIRED)
# tensorrt
set(CUDA_RUNTIME_API_DIRS "/usr/local/cuda-12.9/targets/x86_64-linux/include")

# get *.cpp
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detector/yolox_detector.cpp
)

add_executable(yolox ${SOURCES})

target_compile_options(yolox PRIVATE
    -Wall
    -Wextra
    -Wno-unused-function
)

# include dirs
target_include_directories(yolox PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_RUNTIME_API_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/detector
)
# link libs
target_link_libraries(yolox
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvonnxparser
    nvinfer
)
