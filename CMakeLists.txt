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

# get *.cpp
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detector/yolox_detector.cpp
)

add_executable(yolox ${SOURCES})

# include dirs
target_include_directories(yolox PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/detector
)
# link libs
target_link_libraries(yolox
    ${OpenCV_LIBS}
    nvonnxparser
    nvinfer
    CUDA::cudart
)
