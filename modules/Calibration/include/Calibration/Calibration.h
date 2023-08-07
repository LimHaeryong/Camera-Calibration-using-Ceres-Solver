#ifndef CALIBRATION_H_
#define CALIBRATION_H_

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

class Calibration
{
public:
  Calibration(
    const std::string &image_directory,
    const std::string &distortion_model,
    int corners_width,
    int corners_height,
    double square);
  Calibration(const YAML::Node &config);

  virtual ~Calibration();
  void calibrate();
  void undistort(const cv::Mat &src, cv::Mat &dst);
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

  const uint FIND_CHESSBOARD_FLAGS =
    cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
  const uint CALIB_FISHEYE_FLAGS =
    cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_CHECK_COND + cv::fisheye::CALIB_FIX_SKEW;
  const cv::TermCriteria CORNERSUBPIX_CRITERIA;

  cv::Mat map_x_;
  cv::Mat map_y_;

  double calibrate_implement(
    const std::vector<std::vector<cv::Point3f>> &object_points,
    const std::vector<std::vector<cv::Point2f>> &image_points,
    cv::Size image_size,
    cv::Mat &camera_matrix,
    cv::Mat &dist_coeffs);

  void get_homographies(
    const std::vector<std::vector<cv::Point3f>> &object_points,
    const std::vector<std::vector<cv::Point2f>> &image_points,
    std::vector<cv::Mat> &homographies);

  cv::Mat get_homography(const std::vector<cv::Point3f> &object_point, const std::vector<cv::Point2f> &image_point);

  cv::Mat find_homography(
    const std::vector<cv::Point2f> &src_points,
    const std::vector<cv::Point2f> &dst_points,
    const std::vector<int> &indices);

  cv::Mat find_homography_ransac(
    const std::vector<cv::Point2f> &src_points,
    const std::vector<cv::Point2f> &dst_points,
    int max_iteration = 100,
    double error_threshold = 0.3,
    double confidence = 0.995);

  void refine_homography(
    cv::Mat &H,
    const std::vector<cv::Point2f> &src_points,
    const std::vector<cv::Point2f> &dst_points,
    const std::vector<int> &indices);

  void get_intrinsics(const std::vector<cv::Mat> &homographies, cv::Mat &camera_matrix);

  void compute_v(const cv::Mat &H, int p, int q, cv::Mat &v);

  void get_extrinsics(
    const cv::Mat &camera_matrix,
    const std::vector<cv::Mat> &homographies,
    std::vector<cv::Mat> &extrinsics);

  cv::Mat get_extrinsic(const cv::Mat &camera_matrix, const cv::Mat &homography);

  void refine_rotation_matrix(cv::Mat &R);

  cv::Vec3d get_normalized_coord(const cv::Point3f &object_point, const cv::Mat &extrinsic);

  double refine_all(
    cv::Mat &camera_matrix,
    cv::Mat &dist_coeffs,
    std::vector<cv::Mat> &extrinsics,
    const std::vector<std::vector<cv::Point3f>> &object_points,
    const std::vector<std::vector<cv::Point2f>> &image_points);

  struct projection_error
  {
    projection_error(const cv::Point2f &src_point, const cv::Point2f &dst_point)
    : src_point(src_point), dst_point(dst_point)
    {
    }

    template<typename T>
    bool operator()(const T *const H, T *residuals) const
    {
      T pred_z = H[6] * T(src_point.x) + H[7] * T(src_point.y) + H[8];
      T predict[2];
      predict[0] = (H[0] * T(src_point.x) + H[1] * T(src_point.y) + H[2]) / pred_z;
      predict[1] = (H[3] * T(src_point.x) + H[4] * T(src_point.y) + H[5]) / pred_z;
      residuals[0] = predict[0] - T(dst_point.x);
      residuals[1] = predict[1] - T(dst_point.y);
      return true;
    }

    static ceres::CostFunction *create(const cv::Point2f &src_point, const cv::Point2f &dst_point)
    {
      return (new ceres::AutoDiffCostFunction<projection_error, 2, 9>(new projection_error(src_point, dst_point)));
    }

    const cv::Point2f src_point;
    const cv::Point2f dst_point;
  };

  struct reprojection_error
  {
    reprojection_error(const cv::Point2f &image_point, const cv::Point3f &object_point)
    : image_point(image_point), object_point(object_point)
    {
    }

    template<typename T>
    bool operator()(const T *const intrinsic, const T *const extrinsic, T *residuals) const
    {
      T X[3];
      T object_point_[3];
      object_point_[0] = T(object_point.x);
      object_point_[1] = T(object_point.y);
      object_point_[2] = T(object_point.z);
      ceres::AngleAxisRotatePoint(extrinsic, object_point_, X);
      X[0] += extrinsic[3];
      X[1] += extrinsic[4];
      X[2] += extrinsic[5];

      T x_p = X[0] / X[2];
      T y_p = X[1] / X[2];

      T r = ceres::sqrt(x_p * x_p + y_p * y_p);
      T theta = ceres::atan(r);
      T fisheye_distortion = theta / r *
                             (1.0 + intrinsic[5] * ceres::pow(theta, 2) + intrinsic[6] * ceres::pow(theta, 4) +
                              intrinsic[7] * ceres::pow(theta, 6) + intrinsic[8] * ceres::pow(theta, 8));

      // with skew
      // T predict_x = intrinsic[0] * (fisheye_distortion * x_p + intrinsic[2] *
      // fisheye_distortion * y_p) + intrinsic[3];

      // without skew
      T predict_x = intrinsic[0] * fisheye_distortion * x_p + intrinsic[3];
      T predict_y = intrinsic[1] * fisheye_distortion * y_p + intrinsic[4];

      residuals[0] = predict_x - T(image_point.x);
      residuals[1] = predict_y - T(image_point.y);

      return true;
    }

    static ceres::CostFunction *create(const cv::Point2f &image_point, const cv::Point3f &object_point)
    {
      return (new ceres::AutoDiffCostFunction<reprojection_error, 2, 9, 6>(
        new reprojection_error(image_point, object_point)));
    }

    const cv::Point2f image_point;
    const cv::Point3f object_point;
  };
};

#endif  // CALIBRATION_H_