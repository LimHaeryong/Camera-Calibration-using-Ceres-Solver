#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

using namespace std;

const YAML::Node config_intrinsic = YAML::LoadFile("../config/config_calibration.yaml");

const string IMAGE_DIR = "../extrinsic_images2/";
const string SAVE_DIR = "../output/";
const uint IMAGE_WIDTH = config_intrinsic["image_size"]["width"].as<uint>();
const uint IMAGE_HEIGHT = config_intrinsic["image_size"]["height"].as<uint>();
const bool FISH_EYE = config_intrinsic["fish_eye"].as<bool>();

std::vector<double> camera_matrix_ = config_intrinsic["camera_matrix"].as<std::vector<double>>();
std::vector<double> dist_coeffs_ = config_intrinsic["dist_coeffs"].as<std::vector<double>>();

const cv::Size image_size(IMAGE_WIDTH, IMAGE_HEIGHT);

void undistort_image(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map_x, const cv::Mat& map_y);

int main()
{
  if (cv::utils::fs::exists(SAVE_DIR) == false)
  {
    cv::utils::fs::createDirectories(SAVE_DIR);
  }

  vector<cv::String> image_list;
  cv::glob(IMAGE_DIR + "*.jpg", image_list);
  const uint num_image = image_list.size();

  cv::Mat camera_matrix, dist_coeffs;
  camera_matrix = cv::Mat(3, 3, CV_64F, camera_matrix_.data());
  dist_coeffs = cv::Mat(dist_coeffs_.size(), 1, CV_64F, dist_coeffs_.data());

  cv::Mat new_camera_matrix, map_x, map_y;
  if (FISH_EYE == true)
  {
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(camera_matrix, dist_coeffs, image_size, cv::Mat(),
                                                            new_camera_matrix, 0.0);
    cv::fisheye::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), new_camera_matrix, image_size, CV_32FC1,
                                         map_x, map_y);
  }
  else
  {
    new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, 0.0);
    cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), new_camera_matrix, image_size, CV_32FC1, map_x,
                                map_y);
  }

  cv::namedWindow("src", cv::WINDOW_NORMAL);
  cv::namedWindow("undist", cv::WINDOW_NORMAL);

  for (auto image_path : image_list)
  {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_ANYCOLOR);
    cv::Mat undist;
    undistort_image(image, undist, map_x, map_y);

    string image_name;
    stringstream ss(image_path);
    while (getline(ss, image_name, '/'))
    {
      ;
    }

    cout << SAVE_DIR + "undistorted_" + image_name << endl;
    cv::imwrite(SAVE_DIR + "undistorted_" + image_name, undist);

    cv::waitKey(10);
  }

  return 0;
}

void undistort_image(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map_x, const cv::Mat& map_y)
{
  cv::remap(src, dst, map_x, map_y, cv::INTER_LINEAR);

  cv::imshow("src", src);
  cv::imshow("undist", dst);
}