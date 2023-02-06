#include <opencv2/opencv.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

const YAML::Node config_intrinsic = YAML::LoadFile("../config/config_intrinsic.yaml");
const YAML::Node config = YAML::LoadFile("../config/config_fisheye_2.yaml");

const uint IMAGE_WIDTH = config["image_size"]["width"].as<uint>();
const uint IMAGE_HEIGHT = config["image_size"]["height"].as<uint>();
const bool FISH_EYE = config["fish_eye"].as<bool>();

const double SQUARE_SIZE = config["chessboard"]["square_size"].as<double>();
const uint BOARD_WIDTH = config["chessboard"]["board_width"].as<uint>();
const uint BOARD_HEIGHT = config["chessboard"]["board_height"].as<uint>();

const cv::Size image_size(IMAGE_WIDTH, IMAGE_HEIGHT);
const cv::Size pattern_size(8, 6);

std::vector<double> camera_matrix_ = config_intrinsic["camera_matrix"].as<std::vector<double>>();
std::vector<double> dist_coeffs_ = config_intrinsic["dist_coeffs"].as<std::vector<double>>();

constexpr uint findchessboard_flags =
    cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
constexpr uint calibration_fisheye_flags =
    cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_CHECK_COND + cv::fisheye::CALIB_FIX_SKEW;
const cv::TermCriteria cornersubpix_criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1);

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

  cout << "new_camera_matrix : \n" << new_camera_matrix << endl;

  cv::Mat undist;
  cv::remap(image, undist, map_x, map_y, cv::INTER_LINEAR);

  vector<cv::Point3f> object_point;
  vector<cv::Point2f> image_point;

  cv::Mat image_gray;
  cv::cvtColor(undist, image_gray, cv::COLOR_BGR2GRAY);

  bool ret = cv::findChessboardCorners(image_gray, pattern_size, image_point, findchessboard_flags);
  if (ret == false)
  {
    cout << "failed to find corners!" << endl;
    return -1;
  }

  cv::cornerSubPix(image_gray, image_point, cv::Size(7, 7), cv::Size(-1, -1), cornersubpix_criteria);

  // cv::drawChessboardCorners(undist, pattern_size, image_point, ret);

  object_point.reserve(BOARD_HEIGHT * BOARD_WIDTH);
  for (uint j = 0; j < BOARD_HEIGHT; j++)
  {
    double height = static_cast<double>(j) * SQUARE_SIZE;
    for (uint k = 0; k < BOARD_WIDTH; k++)
    {
      double width = static_cast<double>(k) * SQUARE_SIZE;
      cv::Point3f point(height, width, 0.0);
      object_point.push_back(move(point));
    }
  }

  cv::Mat rvec, tvec;
  cv::solvePnP(object_point, image_point, new_camera_matrix, dist_coeffs, rvec, tvec);

  // cv::drawFrameAxes(undist, new_camera_matrix, dist_coeffs, rvec, tvec,  0.1);
  // cv::namedWindow("undist", cv::WINDOW_NORMAL);
  // cv::imshow("undist", undist);
  // cv::waitKey(0);

  vector<cv::Point2f> homo_object_point;
  homo_object_point.reserve(object_point.size());
  for (auto o : object_point)
  {
    homo_object_point.push_back(cv::Point2f(o.x, o.y));
  }

  cv::Mat H = cv::findHomography(image_point, object_point);

  cout << "H : \n" << H << endl;

  cv::Mat target_image_point = (cv::Mat_<double>(3,1) << image_point[1].x, image_point[1].y, 1.0);
  cout << target_image_point << endl;
  cv::Mat target_object_point = H * target_image_point;
  target_object_point.at<double>(0) /= target_object_point.at<double>(2);
  target_object_point.at<double>(1) /= target_object_point.at<double>(2);
  target_object_point.at<double>(2) = 1.0;

  cout << target_object_point << endl;  
  return 0;
}