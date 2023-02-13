#include <opencv2/opencv.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

const YAML::Node config_intrinsic = YAML::LoadFile("../config/config_calibration_battery.yaml");
const YAML::Node config = YAML::LoadFile("../config/config_xycar_fisheye.yaml");

const uint IMAGE_WIDTH = config["image_size"]["width"].as<uint>();
const uint IMAGE_HEIGHT = config["image_size"]["height"].as<uint>();
const bool FISH_EYE = config["fish_eye"].as<bool>();

const cv::Size image_size(IMAGE_WIDTH, IMAGE_HEIGHT);

std::vector<double> camera_matrix_ = config_intrinsic["camera_matrix"].as<std::vector<double>>();
std::vector<double> dist_coeffs_ = config_intrinsic["dist_coeffs"].as<std::vector<double>>();

constexpr uint findchessboard_flags =
    cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
constexpr uint calibration_fisheye_flags =
    cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_CHECK_COND + cv::fisheye::CALIB_FIX_SKEW;
const cv::TermCriteria cornersubpix_criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1);

int main()
{
  cv::Mat camera_matrix, dist_coeffs;
  camera_matrix = cv::Mat(3, 3, CV_64F, camera_matrix_.data());
  dist_coeffs = cv::Mat(dist_coeffs_.size(), 1, CV_64F, dist_coeffs_.data());

  vector<cv::Point2f> image_point(12);
  image_point[0] = cv::Point2f(113, 291);
  image_point[1] = cv::Point2f(328, 291);
  image_point[2] = cv::Point2f(534, 291);
  image_point[3] = cv::Point2f(205, 259);
  image_point[4] = cv::Point2f(328, 259);
  image_point[5] = cv::Point2f(448, 259);
  image_point[6] = cv::Point2f(242, 246);
  image_point[7] = cv::Point2f(328, 246);
  image_point[8] = cv::Point2f(413, 246);
  image_point[9] = cv::Point2f(262, 239);
  image_point[10] = cv::Point2f(328, 239);
  image_point[11] = cv::Point2f(394, 239);

  vector<cv::Point3f> object_point(12);
  object_point[0] = cv::Point3f(45, -45, 0);
  object_point[1] = cv::Point3f(45, 0, 0);
  object_point[2] = cv::Point3f(45, 45, 0);
  object_point[3] = cv::Point3f(90, -45, 0);
  object_point[4] = cv::Point3f(90, 0, 0);
  object_point[5] = cv::Point3f(90, 45, 0);
  object_point[6] = cv::Point3f(135, -45, 0);
  object_point[7] = cv::Point3f(135, 0, 0);
  object_point[8] = cv::Point3f(135, 45, 0);
  object_point[9] = cv::Point3f(180, -45, 0);
  object_point[10] = cv::Point3f(180, 0, 0);
  object_point[11] = cv::Point3f(180, 45, 0);

  cv::Mat rvec, tvec;
  cv::solvePnP(object_point, image_point, camera_matrix, dist_coeffs, rvec, tvec);

  cout << "rvec : \n" << rvec << endl;
  cout << "tvec : \n" << tvec << endl;

  vector<cv::Point2f> proj_point;
  cv::projectPoints(object_point, rvec, tvec, camera_matrix, dist_coeffs, proj_point);

  for(auto p : proj_point)
  {
    cout << "proj_point : \n" << p << endl;
  }



  return 0;
}