#ifndef CALIBRATION_H_
#define CALIBRATION_H_

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

class Calibration {
public:
  Calibration(const std::string &image_directory,
              const std::string &distortion_model, int corners_width,
              int corners_height, double square);
  Calibration(const YAML::Node &config);

  virtual ~Calibration();
  void calibrate();
  void undistort(const cv::Mat &src, cv::Mat &dst);
  void save_yaml();
  double calibrate_implement(
      const std::vector<std::vector<cv::Point3f>> &object_points,
      const std::vector<std::vector<cv::Point2f>> &image_points,
      cv::Size image_size, cv::Mat &camera_matrix, cv::Mat &dist_coeffs);

private:
  std::vector<std::string> image_paths_;
  bool fisheye_;
  cv::Size image_size_;
  cv::Size pattern_size_;
  double square_;

  cv::Mat camera_matrix_;
  cv::Mat new_camera_matrix_;
  cv::Mat dist_coeffs_;

  const uint FIND_CHESSBOARD_FLAGS = cv::CALIB_CB_ADAPTIVE_THRESH +
                                     cv::CALIB_CB_NORMALIZE_IMAGE +
                                     cv::CALIB_CB_FAST_CHECK;
  const uint CALIB_FISHEYE_FLAGS = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC +
                                   cv::fisheye::CALIB_CHECK_COND +
                                   cv::fisheye::CALIB_FIX_SKEW;
  const cv::TermCriteria CORNERSUBPIX_CRITERIA;

  cv::Mat map_x_;
  cv::Mat map_y_;

  void
  get_homographies(const std::vector<std::vector<cv::Point3f>> &object_points,
                   const std::vector<std::vector<cv::Point2f>> &image_points,
                   std::vector<cv::Mat> &homographies);
  cv::Mat get_homography(const std::vector<cv::Point3f> object_point,
                         const std::vector<cv::Point2f> &image_point);
  void get_intrinsics(const std::vector<cv::Mat> &homographies,
                      cv::Mat &camera_matrix);
  void compute_v(const cv::Mat &H, int p, int q, cv::Mat &v);
  void get_extrinsics(const cv::Mat &camera_matrix,
                      const std::vector<cv::Mat> &homographies,
                      std::vector<cv::Mat> &extrinsics);
  cv::Mat get_extrinsic(const cv::Mat &camera_matrix,
                        const cv::Mat &homography);
  void refine_rotation_matrix(cv::Mat &R);
  void estimate_distortion(
      const std::vector<std::vector<cv::Point3f>> &object_points,
      const std::vector<std::vector<cv::Point2f>> &image_points,
      const std::vector<cv::Mat> &extrinsics, const cv::Mat &camera_matrix,
      cv::Mat &dist_coeffs);
  cv::Vec3d get_normalized_coord(const cv::Point3f &object_point,
                                 const cv::Mat &extrinsic);
  cv::Vec2d get_image_coord(const cv::Vec3d &xy1, const cv::Mat &camera_matrix);
  void refine_all(cv::Mat &camera_matrix, cv::Mat &dist_coeffs,
                  std::vector<cv::Mat> &extrinsics,
                  const std::vector<std::vector<cv::Point3f>> &object_points,
                  const std::vector<std::vector<cv::Point2f>> &image_points);

  struct projection_error {
    projection_error(const cv::Point2f &image_point, const cv::Point3f& object_point)
        : image_point(image_point), object_point(object_point) {}

    template <typename T>
    bool operator()(const T *const intrinsic, const T *const extrinsic,
                    T *residuals) const {
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
      T distortion = theta / r * (T(1.0) + theta * theta * (intrinsic[5] + theta * theta * (intrinsic[6] + theta * theta * (intrinsic[7] + theta * theta * intrinsic[8]))));
      T predict_x = intrinsic[0] * distortion * x_p + intrinsic[2] * distortion * y_p + intrinsic[3];
      T predict_y = intrinsic[1] * distortion * y_p + intrinsic[4];

      residuals[0] = predict_x - T(image_point.x);
      residuals[1] = predict_y - T(image_point.y);
      

      return true;
    }

    static ceres::CostFunction *create(const cv::Point2f &image_point,
                                       const cv::Point3f& object_point) {
      return (new ceres::AutoDiffCostFunction<projection_error, 2, 9, 6>(
          new projection_error(image_point, object_point)));
    }

    const cv::Point2f image_point;
    const cv::Point3f object_point;
  };
};

#endif // CALIBRATION_H_