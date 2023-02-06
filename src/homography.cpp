#include <opencv2/opencv.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

const YAML::Node config = YAML::LoadFile("../config/config_intrinsic.yaml");

const uint IMAGE_WIDTH = config["image_size"]["width"].as<uint>();
const uint IMAGE_HEIGHT = config["image_size"]["height"].as<uint>();
const bool FISH_EYE = config["fish_eye"].as<bool>();

const cv::Size image_size(IMAGE_WIDTH, IMAGE_HEIGHT);

std::vector<double> camera_matrix_ = config["camera_matrix"].as<std::vector<double>>();
std::vector<double> dist_coeffs_ = config["dist_coeffs"].as<std::vector<double>>();

int main()
{
  cv::Mat image = cv::imread("../images_fisheye_2/GOPR3.jpg", cv::IMREAD_COLOR);
  // undistort image
  cv::Mat camera_matrix, dist_coeffs;
  camera_matrix = cv::Mat(3, 3, CV_64F, camera_matrix_.data());
  dist_coeffs = cv::Mat(dist_coeffs_.size(), 1, CV_64F, dist_coeffs_.data());

  cv::Mat new_camera_matrix, map_x, map_y;

  if (FISH_EYE == true)
  {
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(camera_matrix, dist_coeffs, image_size, cv::Mat(),
                                                            new_camera_matrix, 1.0);
    cv::fisheye::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), new_camera_matrix, image_size, CV_32FC1,
                                         map_x, map_y);
  }
  else
  {
    new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, 1.0);
    cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), new_camera_matrix, image_size, CV_32FC1, map_x,
                                map_y);
  }

  cout << "new_camera_matrix : \n" << new_camera_matrix << endl;

  cv::Mat undist;
  cv::remap(image, undist, map_x, map_y, cv::INTER_LINEAR);

  cv::namedWindow("undist", cv::WINDOW_NORMAL);
  cv::imshow("undist", undist);
  cv::waitKey(0);

  return 0;
}