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

Calibration::Calibration(const YAML::Node &config)
{

    std::string distortion_model = config["distortion_model"].as<std::string>();

    if (distortion_model == "fisheye")
    {
        fisheye_ = true;
    }
    else
    {
        fisheye_ = false;
    }

    std::vector<double> camera_matrix = config["camera_matrix"].as<std::vector<double>>();
    std::vector<double> dist_coeffs = config["dist_coeffs"].as<std::vector<double>>();

    image_size_.width = config["image_width"].as<int>();
    image_size_.height = config["image_height"].as<int>();

    camera_matrix_ = cv::Mat(3, 3, CV_64F, camera_matrix.data());
    dist_coeffs_ = cv::Mat(dist_coeffs.size(), 1, CV_64F, dist_coeffs.data());

    if (fisheye_)
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
    std::cout << "camera_matrix : \n"
              << camera_matrix_ << std::endl;
    std::cout << "dist_coeffs : \n"
              << dist_coeffs_ << std::endl;

    if (fisheye_)
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

void Calibration::undistort(const cv::Mat &src, cv::Mat &dst)
{
    cv::remap(src, dst, map_x_, map_y_, cv::INTER_LINEAR);
}

void Calibration::save_yaml()
{
    YAML::Node config;

    if (fisheye_ == true)
    {
        config["distortion_model"] = "fisheye";
    }
    else
    {
        config["distortion_model"] = "optics";
    }

    config["image_width"] = image_size_.width;
    config["image_height"] = image_size_.height;

    std::vector<double> camera_matrix(camera_matrix_.begin<double>(), camera_matrix_.end<double>());
    config["camera_matrix"] = camera_matrix;

    std::vector<double> dist_coeffs(dist_coeffs_.begin<double>(), dist_coeffs_.end<double>());
    config["dist_coeffs"] = dist_coeffs;

    YAML::Emitter emitter;
    emitter << config;
    std::ofstream fout("../result/intrinsic.yaml");
    fout << emitter.c_str();
}

double Calibration::calibrate_implement(const std::vector<std::vector<cv::Point3f>> &object_points, const std::vector<std::vector<cv::Point2f>> &image_points, cv::Size image_size, cv::Mat &camera_matrix, cv::Mat &dist_coeffs)
{
    std::vector<cv::Mat> homographies;
    get_homographies(object_points, image_points, homographies);
    std::vector<cv::Mat> intrinsics;
    get_intrinsics(homographies, intrinsics);


    return 0.0;
}

void Calibration::get_homographies(const std::vector<std::vector<cv::Point3f>> &object_points, const std::vector<std::vector<cv::Point2f>> &image_points, std::vector<cv::Mat>& homographies)
{
    std::size_t M = object_points.size();
    homographies.clear();
    homographies.reserve(M);

    for(size_t i = 0; i < M; ++i)
    {
        cv::Mat homography = get_homography(object_points[i], image_points[i]);
        homographies.push_back(std::move(homography));
    }

}

cv::Mat Calibration::get_homography(const std::vector<cv::Point3f> object_point, const std::vector<cv::Point2f>& image_point)
{
    std::size_t N = object_point.size();
    std::vector<cv::Point2f> object_point_planar(N);
    for(int j = 0; j < N; ++j)
    {
        object_point_planar[j].x = object_point[j].x;
        object_point_planar[j].y = object_point[j].y;
    }

    cv::Mat homography = cv::findHomography(object_point_planar, image_point);
    return homography;
}

void Calibration::get_intrinsics(const std::vector<cv::Mat>& homographies, std::vector<cv::Mat>& intrinsics)
{
    std::size_t M = homographies.size();
    cv::Mat V = cv::Mat::zeros(2 * M, 6, CV_64F);
    for(std::size_t i = 0; i < M; ++i)
    {
        cv::Mat H = homographies[i];
        cv::Mat v0_0 = cv::Mat::zeros(1, 6, CV_64F);
        cv::Mat v0_1 = cv::Mat::zeros(1, 6, CV_64F);
        cv::Mat v1_1 = cv::Mat::zeros(1, 6, CV_64F);
        
        compute_v(H, 0, 0, v0_0);
        compute_v(H, 0, 1, v0_1);
        compute_v(H, 1, 1, v1_1);

        V.row(2 * i) = v0_1;
        V.row(2 * i + 1) = 
    }
}

void Calibration::compute_v(const cv::Mat& H, int p, int q, cv::Mat& v)
{
    v.at<double>(0, 0) = H.at<double>(0, p) * H.at<double>(0, q);
    v.at<double>(0, 1) = H.at<double>(0, p) * H.at<double>(1, q) + H.at<double>(1, p) * H.at<double>(0, q);
    v.at<double>(0, 2) = H.at<double>(1, p) * H.at<double>(1, q);
    v.at<double>(0, 3) = H.at<double>(2, p) * H.at<double>(0, q) + H.at<double>(0, p) * H.at<double>(2, q);
    v.at<double>(0, 4) = H.at<double>(2, p) * H.at<double>(1, q) + H.at<double>(1, p) * H.at<double>(2, q);
    v.at<double>(0, 5) = H.at<double>(2, p) * H.at<double>(2, q);       
}