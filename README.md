# Camera-Calibration
Implementation of camera intrinsic calibration.

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