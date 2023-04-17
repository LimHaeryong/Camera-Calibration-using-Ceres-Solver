# Camera-Calibration using Ceres Solver
This repository contains an implementation of camera intrinsic calibration for optics and fisheye distortion models.

For fisheye distortion models, instead of using the calibrate function in the OpenCV library, I implemented it.

This implementation uses Ceres Solver.

The implementation is based on the ["A Flexible New Technique for Camera Calibration" by Zhang (1999)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf). 

distortion model is based on [opencv fisheye distortion model](https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html)

## Dependencies
- CMake
- OpenCV
- Ceres Solver
- yaml-cpp

## build
```bash
mkdir build && cd build
cmake .. 
make
```

## calibration

```bash
./camera_intrinsic ../resources/checkerboard fisheye 8 6 2.75
```

## save undistorted image

```bash
./save_undist ../resources/checkerboard ../result/intrinsic.yaml
```


## Result

![](./result/22.jpg)
