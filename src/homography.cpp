#include <opencv2/opencv.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

const YAML::Node config_intrinsic = YAML::LoadFile("../config/config_calibration.yaml");
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
  // vector<cv::Point2f> image_point(12);
  // image_point[0] = cv::Point2f(113, 291);
  // image_point[1] = cv::Point2f(328, 291);
  // image_point[2] = cv::Point2f(534, 291);
  // image_point[3] = cv::Point2f(205, 259);
  // image_point[4] = cv::Point2f(328, 259);
  // image_point[5] = cv::Point2f(448, 259);
  // image_point[6] = cv::Point2f(242, 246);
  // image_point[7] = cv::Point2f(328, 246);
  // image_point[8] = cv::Point2f(413, 246);
  // image_point[9] = cv::Point2f(262, 239);
  // image_point[10] = cv::Point2f(328, 239);
  // image_point[11] = cv::Point2f(394, 239);

  vector<cv::Point2f> image_point(12);
  image_point[0] = cv::Point2f(114, 293);
  image_point[1] = cv::Point2f(329, 296);
  image_point[2] = cv::Point2f(537, 297);
  image_point[3] = cv::Point2f(204, 261);
  image_point[4] = cv::Point2f(329, 263);
  image_point[5] = cv::Point2f(449, 266);
  image_point[6] = cv::Point2f(241, 248);
  image_point[7] = cv::Point2f(329, 250);
  image_point[8] = cv::Point2f(415, 252);
  image_point[9] = cv::Point2f(261, 242);
  image_point[10] = cv::Point2f(329, 243);
  image_point[11] = cv::Point2f(395, 244);

  vector<cv::Point2f> homo_object_point(12);
  homo_object_point[0] = cv::Point2f(45, -45);
  homo_object_point[1] = cv::Point2f(45, 0);
  homo_object_point[2] = cv::Point2f(45, 45);
  homo_object_point[3] = cv::Point2f(90, -45);
  homo_object_point[4] = cv::Point2f(90, 0);
  homo_object_point[5] = cv::Point2f(90, 45);
  homo_object_point[6] = cv::Point2f(135, -45);
  homo_object_point[7] = cv::Point2f(135, 0);
  homo_object_point[8] = cv::Point2f(135, 45);
  homo_object_point[9] = cv::Point2f(180, -45);
  homo_object_point[10] = cv::Point2f(180, 0);
  homo_object_point[11] = cv::Point2f(180, 45);

  cv::Mat H = cv::findHomography(image_point, homo_object_point);

  cout << "H : \n"
       << H << endl;

  for (int i = 0; i < 12; i++)
  {
    cv::Mat target_image_point = (cv::Mat_<double>(3, 1) << image_point[i].x, image_point[i].y, 1.0);
    cout << target_image_point << endl;
    cv::Mat target_object_point = H * target_image_point;
    target_object_point.at<double>(0) /= target_object_point.at<double>(2);
    target_object_point.at<double>(1) /= target_object_point.at<double>(2);
    target_object_point.at<double>(2) = 1.0;
    cout << i << " point --------------------------" << endl;
    cout << target_object_point << endl;
  }

  return 0;
}