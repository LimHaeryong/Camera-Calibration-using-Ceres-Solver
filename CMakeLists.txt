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

add_executable(intrinsic
    src/intrinsic.cpp
)
target_link_libraries(intrinsic
    ${OpenCV_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)

add_executable(homography
    src/homography.cpp
)
target_link_libraries(homography
    ${OpenCV_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)

add_executable(save_undistorted_image
    src/save_undistorted_image.cpp
)
target_link_libraries(save_undistorted_image
    ${OpenCV_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)
