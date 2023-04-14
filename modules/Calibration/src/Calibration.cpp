#include "Calibration/Calibration.h"

Calibration::Calibration(const std::string &image_directory, const std::string &distortion_model, int corners_width, int corners_height, double square)
    : square_(square), CORNERSUBPIX_CRITERIA(cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1))
{
    cv::glob(image_directory + "/*.jpg", image_paths_);
    cv::glob(image_directory + "/*.png", image_paths_);

    cv::Mat image = cv::imread(image_paths_[0], cv::IMREAD_GRAYSCALE);
    image_size_ = image.size();

    if (distortion_model == "fisheye")
    {
        fisheye_ = true;
    }
    else
    {
        fisheye_ = false;
    }

    pattern_size_ = cv::Size(corners_width, corners_height);

}

Calibration::~Calibration()
{
}

void Calibration::calibrate()
{
    int num_image = image_paths_.size();
    std::vector<std::vector<cv::Point2f>> image_points;

    image_points.reserve(num_image);

    std::vector<cv::Point2f> corners;
    for (const auto &image_path : image_paths_)
    {
        corners.clear();
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        bool ret = cv::findChessboardCorners(image, pattern_size_, corners, FIND_CHESSBOARD_FLAGS);
        if (ret == false)
        {
            continue;
        }
        cv::cornerSubPix(image, corners, cv::Size(7, 7), cv::Size(-1, -1), CORNERSUBPIX_CRITERIA);
        image_points.push_back(std::move(corners));
    }

    std::vector<cv::Point3f> object_point(pattern_size_.area());
    double height, width;
    for (int i = 0; i < pattern_size_.height; ++i)
    {
        height = static_cast<double>(i) * square_;
        for (int j = 0; j < pattern_size_.width; ++j)
        {
            width = static_cast<double>(j) * square_;
            object_point[i * pattern_size_.width + j] = cv::Point3f(height, width, 0.0);
        }
    }

    std::vector<std::vector<cv::Point3f>> object_points(image_points.size(), object_point);

    std::vector<cv::Mat> rvecs, tvecs;
    double reprojection_error;
    if (fisheye_)
    {
        reprojection_error = cv::fisheye::calibrate(object_points, image_points, image_size_, camera_matrix_, dist_coeffs_, rvecs, tvecs, CALIB_FISHEYE_FLAGS);
    }
    else
    {
        reprojection_error = cv::calibrateCamera(object_points, image_points, image_size_, camera_matrix_, dist_coeffs_, rvecs, tvecs);
    }

    std::cout << "reprojection_error : " << reprojection_error << std::endl;
    std::cout << "camera_matrix : \n" << camera_matrix_ << std::endl;
    std::cout << "dist_coeffs : \n" << dist_coeffs_ << std::endl;

    if(fisheye_)
    {
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(camera_matrix_, dist_coeffs_, image_size_, cv::Mat(), new_camera_matrix_, 0.0);
        cv::fisheye::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat(), new_camera_matrix_, image_size_, CV_32FC1, map_x_, map_y_);
    }
    else
    {
        new_camera_matrix_ = cv::getOptimalNewCameraMatrix(camera_matrix_, dist_coeffs_, image_size_, 0.0);
        cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat(), new_camera_matrix_, image_size_, CV_32FC1, map_x_, map_y_);
    }
}

void Calibration::undistort(const cv::Mat& src, cv::Mat& dst)
{
    cv::remap(src, dst, map_x_, map_y_, cv::INTER_LINEAR);
}

void Calibration::save_yaml()
{
    YAML::Node config;
    std::vector<double> camera_matrix(camera_matrix_.begin<double>(), camera_matrix_.end<double>());
    config["camera_matrix"] = camera_matrix;

    std::vector<double> dist_coeffs(dist_coeffs_.begin<double>(), dist_coeffs_.end<double>());
    config["dist_coeffs"] = dist_coeffs;

    YAML::Emitter emitter;
    emitter << config;
    std::ofstream fout("intrinsic.yaml");
    fout << emitter.c_str();
}