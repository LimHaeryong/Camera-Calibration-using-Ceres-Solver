cmake_minimum_required(VERSION 3.0.0)
project(CAMERA-CALIBRATION LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(modules)

add_executable(camera_intrinsic 
    examples/camera_intrinsic.cpp
)
target_link_libraries(camera_intrinsic PRIVATE
    calibration_module
)

add_executable(save_undist 
    examples/save_undistorted_images.cpp
)
target_link_libraries(save_undist PRIVATE
    calibration_module
)
