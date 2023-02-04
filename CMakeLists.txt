cmake_minimum_required(VERSION 3.0.0)

add_compile_options(-std=c++14)

project(CAMERA-CALIBRATION)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_definitions(
    ${PCL_DEFINITIONS}
)

link_directories(
    ${OpenCV_LIBRARY_DIRS}
)

add_executable(calibration
    src/calibration.cpp
)
target_link_libraries(calibration
    ${OpenCV_LIBRARIES}
)
