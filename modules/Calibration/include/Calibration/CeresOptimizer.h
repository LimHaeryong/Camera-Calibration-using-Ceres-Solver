#ifndef CALIBRATION_CERES_OPTIMIZER_H_
#define CALIBRATION_CERES_OPTIMIZER_H_

#include <vector>

#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

class CeresOptimizer
{
public:
    enum Type
    {
        homography_geometric,
        distortion_geometric
    };

    CeresOptimizer(Type optimizer_type) : optimizer_type(optimizer_type)
    {
        problem = std::make_unique<ceres::Problem>();
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.num_threads = 8;
        options.minimizer_progress_to_stdout = false;

        switch(optimizer_type)
        {
          case Type::homography_geometric :
            options.parameter_tolerance = 1e-12;
            options.gradient_tolerance = 1e-14;
            options.function_tolerance = 1e-10;
            break;
        }
        
    }

    virtual ~CeresOptimizer() {}

    void clear()
    {
        problem.reset(nullptr);
        problem = std::make_unique<ceres::Problem>();
    }

    void solve();

    void add_residual(const cv::Point2f& image_point, const cv::Point3f& object_point, std::vector<double>& intrinsic, std::vector<double>& extrinsic);
    void add_residual(const cv::Point2f& src_point, const cv::Point2f& dst_point, std::vector<double>& homography);

private:
    std::unique_ptr<ceres::Problem> problem;
    ceres::Solver::Options options;
    Type optimizer_type;

};

struct homography_reproj_error
{
  homography_reproj_error(const cv::Point2f &src_point, const cv::Point2f &dst_point)
  : src_point(src_point), dst_point(dst_point)
  {
  }
  template<typename T>
  bool operator()(const T *const H, T *residuals) const
  {
    T pred_z = H[6] * T(src_point.x) + H[7] * T(src_point.y) + H[8];
    T dst_point_est[2];
    dst_point_est[0] = (H[0] * T(src_point.x) + H[1] * T(src_point.y) + H[2]) / pred_z;
    dst_point_est[1] = (H[3] * T(src_point.x) + H[4] * T(src_point.y) + H[5]) / pred_z;
    residuals[0] = dst_point_est[0] - T(dst_point.x);
    residuals[1] = dst_point_est[1] - T(dst_point.y);
    return true;
  }
  static ceres::CostFunction *create(const cv::Point2f &src_point, const cv::Point2f &dst_point)
  {
    return (new ceres::AutoDiffCostFunction<homography_reproj_error, 2, 9>(new homography_reproj_error(src_point, dst_point)));
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

#endif // CALIBRATION_CERES_OPTIMIZER_H_