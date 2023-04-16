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
    projection_error(const cv::Point2d &image_point, const double *object_point)
        : image_point(image_point), object_point(object_point) {}

    template <typename T>
    bool operator()(const T *const intrinsic, const T *const extrinsic,
                    T *residuals) const {
      T X[3];
      ceres::AngleAxisRotatePoint(extrinsic, object_point, X);
      X[0] += extrinsic[3];
      X[1] += extrinsic[4];
      X[2] += extrinsic[5];

      T x_p = intrinsic[0] * X[0] / X[2] + intrinsic[2] * X[1] / X[2] +
              intrinsic[3];
      T y_p = intrinsic[1] * X[1] / X[2] + intrinsic[4];

      T r_2 = x_p * x_p + y_p * y_p;
      T r_4 = r_2 * r_2;
      T r_6 = r_2 * r_2 * r_2;
      T r_8 = r_2 * r_2 * r_2 * r_2;
      T x_c = x_p * (1 + r_2 * intrinsic[5] + r_4 * intrinsic[6] +
                     r_6 * intrinsic[7] + r_8 * intrinsic[8]);
      T y_c = y_p * (1 + r_2 * intrinsic[5] + r_4 * intrinsic[6] +
                     r_6 * intrinsic[7] + r_8 * intrinsic[8]);

      residuals[0] = T(image_point.x) - x_c;
      residuals[1] = T(image_point.y) - y_c;
      return true;
    }

    static ceres::CostFunction *create(const cv::Point2d &image_point,
                                       const double *object_point) {
      return (new ceres::AutoDiffCostFunction<projection_error, 2, 9, 6>(
          new projection_error(image_point, object_point)));
    }

    const cv::Point2d image_point;
    const double *object_point;
  };
};

#endif // CALIBRATION_H_