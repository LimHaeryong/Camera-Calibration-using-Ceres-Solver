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
    Calibration(const std::string& image_directory, const std::string& distortion_model, int corners_width, int corners_height, double square);
    Calibration(const YAML::Node& config);

    virtual ~Calibration();
    void calibrate();
    void undistort(const cv::Mat& src, cv::Mat& dst);
    void save_yaml();

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
};


#endif // CALIBRATION_H_