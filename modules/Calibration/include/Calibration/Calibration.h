#ifndef CALIBRATION_H_
#define CALIBRATION_H_

#include <fstream>
#include <string>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

class Calibration
{
public:
    Calibration(const std::string &image_directory, const std::string &distortion_model, int corners_width, int corners_height, double square);
    Calibration(const YAML::Node &config);

    virtual ~Calibration();
    void calibrate();
    void undistort(const cv::Mat &src, cv::Mat &dst);
    void save_yaml();
    double calibrate_implement(const std::vector<std::vector<cv::Point3f>> &object_points, const std::vector<std::vector<cv::Point2f>> &image_points, cv::Size image_size, cv::Mat &camera_matrix, cv::Mat &dist_coeffs);

private:
    std::vector<std::string> image_paths_;
    bool fisheye_;
    cv::Size image_size_;
    cv::Size pattern_size_;
    double square_;

    cv::Mat camera_matrix_;
    cv::Mat new_camera_matrix_;
    cv::Mat dist_coeffs_;

    const uint FIND_CHESSBOARD_FLAGS = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
    const uint CALIB_FISHEYE_FLAGS = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_CHECK_COND + cv::fisheye::CALIB_FIX_SKEW;
    const cv::TermCriteria CORNERSUBPIX_CRITERIA;

    cv::Mat map_x_;
    cv::Mat map_y_;

    void get_homographies(const std::vector<std::vector<cv::Point3f>> &object_points, const std::vector<std::vector<cv::Point2f>> &image_points, std::vector<cv::Mat> &homographies);
    cv::Mat get_homography(const std::vector<cv::Point3f> object_point, const std::vector<cv::Point2f> &image_point);
    void get_intrinsics(const std::vector<cv::Mat> &homographies, cv::Mat &camera_matrix);
    void compute_v(const cv::Mat &H, int p, int q, cv::Mat &v);
    void get_extrinsics(const cv::Mat& camera_matrix, const std::vector<cv::Mat>& homographies, std::vector<cv::Mat>& extrinsics);
    cv::Mat get_extrinsic(const cv::Mat& camera_matrix, const cv::Mat& homography);
    void refine_rotation_matrix(cv::Mat& R);

};

#endif // CALIBRATION_H_