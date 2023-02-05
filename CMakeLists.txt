cmake_minimum_required(VERSION 3.0.0)

add_compile_options(-std=c++14)

project(CAMERA-CALIBRATION)

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

add_definitions(
    ${OpenCV_DEFINITIONS}
    ${YAMP_CPP_DEFINITIONS}
)

link_directories(
    ${OpenCV_LIBRARY_DIRS}
    ${YAML_CPP_LIBRARY_DIRS}
)

add_executable(calibration
    src/calibration.cpp
)
target_link_libraries(calibration
    ${OpenCV_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)
