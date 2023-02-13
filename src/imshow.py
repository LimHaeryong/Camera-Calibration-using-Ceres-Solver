import cv2

image = cv2.imread('/home/haeryong/Desktop/depth/Camera-Calibration-main/undistort_extrinsic_images2/undistorted_data_1676002168.660921.jpg')

cv2.namedWindow("src", cv2.WINDOW_NORMAL)

cv2.imshow("src", image)

while True:
    if(cv2.waitKey(0) == 'q'):
        break
