#include <opencv2/opencv.hpp>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

// const YAML::Node config = YAML::LoadFile("../config/config.yaml");
//const YAML::Node config = YAML::LoadFile("../config/config_fisheye.yaml");
const YAML::Node config = YAML::LoadFile("../config/config_fisheye_2.yaml");

const string IMAGE_DIR = config["image_dir"].as<string>();
const uint IMAGE_WIDTH = config["image_size"]["width"].as<uint>();
const uint IMAGE_HEIGHT = config["image_size"]["height"].as<uint>();
const bool FISH_EYE = config["fish_eye"].as<bool>();

const double SQUARE_SIZE = config["chessboard"]["square_size"].as<double>();
const uint BOARD_WIDTH = config["chessboard"]["board_width"].as<uint>();
const uint BOARD_HEIGHT = config["chessboard"]["board_height"].as<uint>();

const cv::Size image_size(IMAGE_WIDTH, IMAGE_HEIGHT);
const cv::Size pattern_size(BOARD_WIDTH, BOARD_HEIGHT);

constexpr uint findchessboard_flags =
    cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
constexpr uint calibration_fisheye_flags =
    cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_CHECK_COND + cv::fisheye::CALIB_FIX_SKEW;
const cv::TermCriteria cornersubpix_criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1);

void undistort_image(const cv::Mat& src, const cv::Mat& map_x, const cv::Mat& map_y);
void compute_intrinsic(cv::Mat& camera_matrix, cv::Mat& dist_coeffs);
void save_intrinsic(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);

int main()
{
  vector<cv::String> image_list;
  cv::glob(IMAGE_DIR + "*.jpg", image_list);
  const uint num_image = image_list.size();

  cv::Mat camera_matrix, dist_coeffs;
  compute_intrinsic(camera_matrix, dist_coeffs);

  save_intrinsic(camera_matrix, dist_coeffs);

  // undistort images
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

  // save intrinsic parameters

  cv::namedWindow("src", cv::WINDOW_NORMAL);
  cv::namedWindow("undist", cv::WINDOW_NORMAL);

  for (auto image_path : image_list)
  {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    undistort_image(image, map_x, map_y);
  }
  return 0;
}

void compute_intrinsic(cv::Mat& camera_matrix, cv::Mat& dist_coeffs)
{
  vector<cv::String> image_list;
  cv::glob(IMAGE_DIR + "*.jpg", image_list);
  const uint num_image = image_list.size();

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

    bool ret = cv::findChessboardCorners(image_gray, pattern_size, corners, findchessboard_flags);
    if (ret == false)
    {
      continue;
    }
    cv::cornerSubPix(image_gray, corners, cv::Size(7, 7), cv::Size(-1, -1), cornersubpix_criteria);
    image_points.push_back(move(corners));
  }

  for (uint i = 0; i < image_points.size(); i++)
  {
    vector<cv::Point3f> object_point;
    object_point.reserve(BOARD_HEIGHT * BOARD_WIDTH);
    for (uint j = 0; j < BOARD_HEIGHT; j++)
    {
      double height = static_cast<double>(j) * SQUARE_SIZE;
      for (uint k = 0; k < BOARD_WIDTH; k++)
      {
        double width = static_cast<double>(k) * SQUARE_SIZE;
        cv::Point3f point(height, width, 0.f);
        object_point.push_back(move(point));
      }
    }
    object_points.push_back(move(object_point));
  }

  vector<cv::Mat> rvecs, tvecs;
  double reprojection_error;
  if (FISH_EYE == true)
  {
    reprojection_error = cv::fisheye::calibrate(object_points, image_points, image_size, camera_matrix, dist_coeffs,
                                                rvecs, tvecs, calibration_fisheye_flags);
  }
  else
  {
    reprojection_error =
        cv::calibrateCamera(object_points, image_points, image_size, camera_matrix, dist_coeffs, rvecs, tvecs);
  }

  cout << "reprojection_error : " << reprojection_error << endl;
  cout << "camera_matrix : \n" << camera_matrix << endl;
  cout << "dist_coeffs : \n" << dist_coeffs << endl;
}

void save_intrinsic(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
{
  YAML::Node config_intrinsic;
  // config_intrinsic["image_size"] = YAML::Node(YAML::NodeType::Map);
  config_intrinsic["image_size"]["width"] = IMAGE_WIDTH;
  config_intrinsic["image_size"]["height"] = IMAGE_HEIGHT;
  config_intrinsic["fish_eye"] = FISH_EYE;

  std::vector<double> camera_matrix_(camera_matrix.begin<double>(), camera_matrix.end<double>());
  config_intrinsic["camera_matrix"] = camera_matrix_;

  std::vector<double> dist_coeffs_(dist_coeffs.begin<double>(), dist_coeffs.end<double>());
  config_intrinsic["dist_coeffs"] = dist_coeffs_;

  YAML::Emitter emitter;
  emitter << config_intrinsic;
  ofstream fout("../config/config_intrinsic.yaml");
  fout << emitter.c_str();
}

void undistort_image(const cv::Mat& src, const cv::Mat& map_x, const cv::Mat& map_y)
{
  cv::Mat undist;
  cv::remap(src, undist, map_x, map_y, cv::INTER_LINEAR);

  cv::imshow("src", src);
  cv::imshow("undist", undist);

  cv::waitKey(0);
}