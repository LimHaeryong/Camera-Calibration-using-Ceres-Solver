#include <opencv2/opencv.hpp>

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <vector>

using namespace std;

const YAML::Node config = YAML::LoadFile("../config.yaml");

const string IMAGE_DIR = config["image_dir"].as<string>();
const float SQUARE_SIZE = config["chessboard"]["square_size"].as<float>();
const uint BOARD_WIDTH = config["chessboard"]["board_width"].as<uint>();
const uint BOARD_HEIGHT = config["chessboard"]["board_height"].as<uint>();
const bool FISH_EYE = config["fish_eye"].as<bool>();

int main()
{
  uint flags;
  if (FISH_EYE == false)
  {
    flags = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
  }
  else
  {
    flags = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
  }

  vector<cv::String> image_list;
  cv::glob(IMAGE_DIR + "*.jpg", image_list);
  const uint num_image = image_list.size();

  cv::Mat sample_image = cv::imread(image_list[0], cv::IMREAD_GRAYSCALE);
  const cv::Size image_size = sample_image.size();
  const cv::Size pattern_size(BOARD_WIDTH, BOARD_HEIGHT);

  vector<vector<cv::Point3f>> object_points;
  vector<vector<cv::Point2f>> image_points;
  vector<cv::Point2f> corners;
  object_points.reserve(num_image);
  image_points.reserve(num_image);

  for (auto image_path : image_list)
  {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

    bool ret = cv::findChessboardCorners(image_gray, pattern_size, corners, flags);
    if (ret == false)
    {
      continue;
    }
    image_points.push_back(move(corners));
  }

  for (uint i = 0; i < image_points.size(); i++)
  {
    vector<cv::Point3f> object_point;
    object_point.reserve(BOARD_HEIGHT * BOARD_WIDTH);
    for (uint j = 0; j < BOARD_HEIGHT; j++)
    {
      float height = static_cast<float>(j) * SQUARE_SIZE;
      for (uint k = 0; k < BOARD_WIDTH; k++)
      {
        float width = static_cast<float>(k) * SQUARE_SIZE;
        cv::Point3f point(height, width, 0.f);
        object_point.push_back(move(point));
      }
    }
    object_points.push_back(move(object_point));
  }

  cv::Mat camera_matrix, dist_coeffs;
  vector<cv::Mat> rvecs, tvecs;
  double reprojection_error =
      cv::calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs, rvecs, tvecs);

  cout << "reprojection_error : " << reprojection_error << endl;
  cout << "camera_matrix : \n" << camera_matrix << endl;
  cout << "dist_coeffs : \n" << dist_coeffs << endl;

  // undistort images
  cv::Mat map_x, map_y;
  cv::initUndistortRectifyMap(camera_matrix, dist_coeffs, cv::Mat(), cv::Mat(), image_size, CV_32FC1, map_x, map_y);

  for (auto image_path : image_list)
  {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat undist;
    cv::remap(image, undist, map_x, map_y, cv::INTER_LINEAR);
    cv::imshow("src", image);
    cv::imshow("undist", undist);

    cv::waitKey(1000);
  }
  return 0;
}