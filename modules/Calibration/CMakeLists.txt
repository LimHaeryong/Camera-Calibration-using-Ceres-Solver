cmake_minimum_required(VERSION 3.0.0)
project(CAMERA-CALIBRATION LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(calibration_module
    src/Calibration.cpp
)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV Found! - ${OpenCV_DIR}")
endif()

find_package(yaml-cpp REQUIRED)
if(yaml-cpp_FOUND)
    message(STATUS "yaml-cpp Found! - ${YAML_CPP_DIR}")
endif()

target_include_directories(calibration_module PUBLIC
    include
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

target_link_libraries(calibration_module PUBLIC
    ${OpenCV_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)