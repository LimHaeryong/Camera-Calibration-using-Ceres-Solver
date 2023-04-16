#include "Calibration/Calibration.h"

Calibration::Calibration(const std::string &image_directory,
                         const std::string &distortion_model, int corners_width,
                         int corners_height, double square)
    : square_(square),
      CORNERSUBPIX_CRITERIA(cv::TermCriteria(
          cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1)) {
  cv::glob(image_directory + "/*.jpg", image_paths_);
  cv::glob(image_directory + "/*.png", image_paths_);

  cv::Mat image = cv::imread(image_paths_[0], cv::IMREAD_GRAYSCALE);
  image_size_ = image.size();

  if (distortion_model == "fisheye") {
    fisheye_ = true;
  } else {
    fisheye_ = false;
  }

  pattern_size_ = cv::Size(corners_width, corners_height);
}

Calibration::Calibration(const YAML::Node &config) {

  std::string distortion_model = config["distortion_model"].as<std::string>();

  if (distortion_model == "fisheye") {
    fisheye_ = true;
  } else {
    fisheye_ = false;
  }

  std::vector<double> camera_matrix =
      config["camera_matrix"].as<std::vector<double>>();
  std::vector<double> dist_coeffs =
      config["dist_coeffs"].as<std::vector<double>>();

  image_size_.width = config["image_width"].as<int>();
  image_size_.height = config["image_height"].as<int>();

  camera_matrix_ = cv::Mat(3, 3, CV_64F, camera_matrix.data());
  dist_coeffs_ = cv::Mat(dist_coeffs.size(), 1, CV_64F, dist_coeffs.data());

  if (fisheye_) {
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        camera_matrix_, dist_coeffs_, image_size_, cv::Mat(),
        new_camera_matrix_, 0.0);
    cv::fisheye::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
                                         cv::Mat(), new_camera_matrix_,
                                         image_size_, CV_32FC1, map_x_, map_y_);
  } else {
    new_camera_matrix_ = cv::getOptimalNewCameraMatrix(
        camera_matrix_, dist_coeffs_, image_size_, 0.0);
    cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat(),
                                new_camera_matrix_, image_size_, CV_32FC1,
                                map_x_, map_y_);
  }
}

Calibration::~Calibration() {}

void Calibration::calibrate() {
  int num_image = image_paths_.size();
  std::vector<std::vector<cv::Point2f>> image_points;

  image_points.reserve(num_image);

  std::vector<cv::Point2f> corners;
  for (const auto &image_path : image_paths_) {
    corners.clear();
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    bool ret = cv::findChessboardCorners(image, pattern_size_, corners,
                                         FIND_CHESSBOARD_FLAGS);
    if (ret == false) {
      continue;
    }
    cv::cornerSubPix(image, corners, cv::Size(7, 7), cv::Size(-1, -1),
                     CORNERSUBPIX_CRITERIA);
    image_points.push_back(std::move(corners));
  }

  std::vector<cv::Point3f> object_point(pattern_size_.area());
  double height, width;
  for (int i = 0; i < pattern_size_.height; ++i) {
    height = static_cast<double>(i) * square_;
    for (int j = 0; j < pattern_size_.width; ++j) {
      width = static_cast<double>(j) * square_;
      object_point[i * pattern_size_.width + j] =
          cv::Point3f(height, width, 0.0);
    }
  }

  std::vector<std::vector<cv::Point3f>> object_points(image_points.size(),
                                                      object_point);

  std::vector<cv::Mat> rvecs, tvecs;
  double reprojection_error;
  if (fisheye_) {
    reprojection_error = calibrate_implement(
        object_points, image_points, image_size_, camera_matrix_, dist_coeffs_);
    return; // for debug
    reprojection_error = cv::fisheye::calibrate(
        object_points, image_points, image_size_, camera_matrix_, dist_coeffs_,
        rvecs, tvecs, CALIB_FISHEYE_FLAGS);
  } else {
    reprojection_error =
        cv::calibrateCamera(object_points, image_points, image_size_,
                            camera_matrix_, dist_coeffs_, rvecs, tvecs);
  }

  std::cout << "reprojection_error : " << reprojection_error << std::endl;
  std::cout << "camera_matrix : \n" << camera_matrix_ << std::endl;
  std::cout << "dist_coeffs : \n" << dist_coeffs_ << std::endl;

  if (fisheye_) {
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        camera_matrix_, dist_coeffs_, image_size_, cv::Mat(),
        new_camera_matrix_, 0.0);
    cv::fisheye::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
                                         cv::Mat(), new_camera_matrix_,
                                         image_size_, CV_32FC1, map_x_, map_y_);
  } else {
    new_camera_matrix_ = cv::getOptimalNewCameraMatrix(
        camera_matrix_, dist_coeffs_, image_size_, 0.0);
    cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat(),
                                new_camera_matrix_, image_size_, CV_32FC1,
                                map_x_, map_y_);
  }
}

void Calibration::undistort(const cv::Mat &src, cv::Mat &dst) {
  cv::remap(src, dst, map_x_, map_y_, cv::INTER_LINEAR);
}

void Calibration::save_yaml() {
  YAML::Node config;

  if (fisheye_ == true) {
    config["distortion_model"] = "fisheye";
  } else {
    config["distortion_model"] = "optics";
  }

  config["image_width"] = image_size_.width;
  config["image_height"] = image_size_.height;

  std::vector<double> camera_matrix(camera_matrix_.begin<double>(),
                                    camera_matrix_.end<double>());
  config["camera_matrix"] = camera_matrix;

  std::vector<double> dist_coeffs(dist_coeffs_.begin<double>(),
                                  dist_coeffs_.end<double>());
  config["dist_coeffs"] = dist_coeffs;

  YAML::Emitter emitter;
  emitter << config;
  std::ofstream fout("../result/intrinsic.yaml");
  fout << emitter.c_str();
}

double Calibration::calibrate_implement(
    const std::vector<std::vector<cv::Point3f>> &object_points,
    const std::vector<std::vector<cv::Point2f>> &image_points,
    cv::Size image_size, cv::Mat &camera_matrix, cv::Mat &dist_coeffs) {
  std::vector<cv::Mat> homographies, extrinsics;
  get_homographies(object_points, image_points, homographies);
  get_intrinsics(homographies, camera_matrix);
  get_extrinsics(camera_matrix, homographies, extrinsics);
  estimate_distortion(object_points, image_points, extrinsics, camera_matrix,
                      dist_coeffs);
  refine_all(camera_matrix, dist_coeffs, extrinsics, object_points,
             image_points);
  return 0.0;
}

void Calibration::get_homographies(
    const std::vector<std::vector<cv::Point3f>> &object_points,
    const std::vector<std::vector<cv::Point2f>> &image_points,
    std::vector<cv::Mat> &homographies) {
  std::size_t M = object_points.size();
  homographies.clear();
  homographies.reserve(M);

  for (size_t i = 0; i < M; ++i) {
    cv::Mat homography = get_homography(object_points[i], image_points[i]);
    homographies.push_back(std::move(homography));
  }
}

cv::Mat
Calibration::get_homography(const std::vector<cv::Point3f> object_point,
                            const std::vector<cv::Point2f> &image_point) {
  std::size_t N = object_point.size();
  std::vector<cv::Point2f> object_point_planar(N);
  for (int j = 0; j < N; ++j) {
    object_point_planar[j].x = object_point[j].x;
    object_point_planar[j].y = object_point[j].y;
  }
  cv::Mat homography = cv::findHomography(object_point_planar, image_point);
  return homography;
}

void Calibration::get_intrinsics(const std::vector<cv::Mat> &homographies,
                                 cv::Mat &camera_matrix) {
  std::size_t M = homographies.size();
  cv::Mat V = cv::Mat::zeros(2 * M, 6, CV_64F);
  for (std::size_t i = 0; i < M; ++i) {
    cv::Mat H = homographies[i];

    cv::Mat v0_0 = cv::Mat::zeros(1, 6, CV_64F);
    cv::Mat v0_1 = cv::Mat::zeros(1, 6, CV_64F);
    cv::Mat v1_1 = cv::Mat::zeros(1, 6, CV_64F);

    compute_v(H, 0, 0, v0_0);
    compute_v(H, 0, 1, v0_1);
    compute_v(H, 1, 1, v1_1);

    v0_1.copyTo(V.row(2 * i));
    V.row(2 * i + 1) = v0_0 - v1_1;
  }

  cv::SVD svd(V, cv::SVD::FULL_UV);
  cv::Vec6d b = svd.vt.row(svd.vt.rows - 1);

  camera_matrix = cv::Mat::zeros(3, 3, CV_64F);
  double w = b[0] * b[2] * b[5] - b[1] * b[1] * b[5] - b[0] * b[4] * b[4] +
             2 * b[1] * b[3] * b[4] - b[2] * b[3] * b[3];
  double d = b[0] * b[2] - b[1] * b[1];
  camera_matrix.at<double>(0, 0) = sqrt(w / (d * b[0]));
  camera_matrix.at<double>(1, 1) = sqrt(w / (d * d) * b[0]);
  camera_matrix.at<double>(0, 1) = sqrt(w / (d * d * b[0])) * b[1];
  camera_matrix.at<double>(0, 2) = (b[1] * b[4] - b[2] * b[3]) / d;
  camera_matrix.at<double>(1, 2) = (b[1] * b[3] - b[0] * b[4]) / d;
  camera_matrix.at<double>(2, 2) = 1.0;
}

void Calibration::compute_v(const cv::Mat &H, int p, int q, cv::Mat &v) {
  v.at<double>(0, 0) = H.at<double>(0, p) * H.at<double>(0, q);
  v.at<double>(0, 1) = H.at<double>(0, p) * H.at<double>(1, q) +
                       H.at<double>(1, p) * H.at<double>(0, q);
  v.at<double>(0, 2) = H.at<double>(1, p) * H.at<double>(1, q);
  v.at<double>(0, 3) = H.at<double>(2, p) * H.at<double>(0, q) +
                       H.at<double>(0, p) * H.at<double>(2, q);
  v.at<double>(0, 4) = H.at<double>(2, p) * H.at<double>(1, q) +
                       H.at<double>(1, p) * H.at<double>(2, q);
  v.at<double>(0, 5) = H.at<double>(2, p) * H.at<double>(2, q);
}

void Calibration::get_extrinsics(const cv::Mat &camera_matrix,
                                 const std::vector<cv::Mat> &homographies,
                                 std::vector<cv::Mat> &extrinsics) {
  std::size_t M = homographies.size();
  extrinsics.clear();
  extrinsics.reserve(M);
  for (std::size_t i = 0; i < M; ++i) {
    cv::Mat extrinsic = get_extrinsic(camera_matrix, homographies[i]);
    extrinsics.push_back(std::move(extrinsic));
  }
}

cv::Mat Calibration::get_extrinsic(const cv::Mat &camera_matrix,
                                   const cv::Mat &homography) {
  cv::Mat A_inv = camera_matrix.inv();
  cv::Mat R(3, 3, CV_64F);
  homography.col(0);
  double lambda = 1.0 / cv::norm(A_inv * homography.col(0));
  R.col(0) = lambda * A_inv * homography.col(0);
  R.col(1) = lambda * A_inv * homography.col(1);
  cv::Mat r2 = R.col(0).cross(R.col(1));
  r2.copyTo(R.col(2));
  refine_rotation_matrix(R);

  cv::Mat t = lambda * A_inv * homography.col(2);
  cv::Mat extrinsic(3, 4, CV_64F);
  R.copyTo(extrinsic.colRange(0, 3));
  t.copyTo(extrinsic.col(3));

  return extrinsic;
}

void Calibration::refine_rotation_matrix(cv::Mat &R) {
  cv::Mat S, U, Vt;
  cv::SVD::compute(R, S, U, Vt);
  R = U * Vt;
}

void Calibration::estimate_distortion(
    const std::vector<std::vector<cv::Point3f>> &object_points,
    const std::vector<std::vector<cv::Point2f>> &image_points,
    const std::vector<cv::Mat> &extrinsics, const cv::Mat &camera_matrix,
    cv::Mat &dist_coeffs) {
  int M = extrinsics.size();
  int N = object_points.size();

  double u_c = camera_matrix.at<double>(0, 2);
  double v_c = camera_matrix.at<double>(1, 2);

  cv::Mat D(2 * M * N, 4, CV_64F);
  cv::Mat d(2 * M * N, 1, CV_64F);
  int l = 0;
  for (int i = 0; i < M; ++i) {
    cv::Mat extrinsic = extrinsics[i];
    for (int j = 0; j < N; ++j) {
      cv::Vec3d xy1 = get_normalized_coord(object_points[i][j], extrinsic);
      double r_2 = xy1[0] * xy1[0] + xy1[1] * xy1[1];
      double r_4 = r_2 * r_2;
      double r_6 = r_2 * r_2 * r_2;
      double r_8 = r_2 * r_2 * r_2 * r_2;

      cv::Mat uv = camera_matrix.rowRange(0, 2) * xy1;
      double d_u = uv.at<double>(0, 0) - u_c;
      double d_v = uv.at<double>(1, 0) - v_c;

      D.row(2 * l) = (cv::Mat_<double>(1, 4) << r_2 * d_u, r_4 * d_u, r_6 * d_u,
                      r_8 * d_u);
      D.row(2 * l + 1) = (cv::Mat_<double>(1, 4) << r_2 * d_v, r_4 * d_v,
                          r_6 * d_v, r_8 * d_v);

      d.at<double>(2 * l, 0) = image_points[i][j].x - uv.at<double>(0, 0);
      d.at<double>(2 * l + 1, 0) = image_points[i][j].y - uv.at<double>(1, 0);
      ++l;
    }
  }

  bool ret = cv::solve(D, d, dist_coeffs, cv::DECOMP_SVD);
}

cv::Vec3d Calibration::get_normalized_coord(const cv::Point3f &object_point,
                                            const cv::Mat &extrinsic) {
  cv::Vec4d X;
  X[0] = object_point.x;
  X[1] = object_point.y;
  X[2] = object_point.z;
  X[3] = 1.0;

  cv::Mat ret = extrinsic * X;
  ret *= 1.0 / ret.at<double>(2, 0);

  return cv::Vec3d(ret);
}

void Calibration::refine_all(
    cv::Mat &camera_matrix, cv::Mat &dist_coeffs,
    std::vector<cv::Mat> &extrinsics,
    const std::vector<std::vector<cv::Point3f>> &object_points,
    const std::vector<std::vector<cv::Point2f>> &image_points) {

  std::size_t M = image_points.size();
  std::size_t N = image_points[0].size();

  double *parameters = new double[9];
  double **extrinsic_array = new double *[M];
  for(int i = 0; i < M; ++i)
  {
    extrinsic_array[i] = new double[6];
  }

  parameters[0] = camera_matrix.at<double>(0, 0);
  parameters[1] = camera_matrix.at<double>(1, 1);
  parameters[2] = camera_matrix.at<double>(0, 1);
  parameters[3] = camera_matrix.at<double>(0, 2);
  parameters[4] = camera_matrix.at<double>(1, 2);
  parameters[5] = dist_coeffs.at<double>(0, 0);
  parameters[6] = dist_coeffs.at<double>(1, 0);
  parameters[7] = dist_coeffs.at<double>(2, 0);
  parameters[8] = dist_coeffs.at<double>(3, 0);

  for (std::size_t i = 0; i < M; ++i) {
    cv::Mat rot = extrinsics[i].colRange(0, 3);
    cv::Mat rvec;
    cv::Rodrigues(rot, rvec);
    extrinsic_array[i][0] = rvec.at<double>(0, 0);
    extrinsic_array[i][1] = rvec.at<double>(1, 0);
    extrinsic_array[i][2] = rvec.at<double>(2, 0);
    extrinsic_array[i][3] = extrinsics[i].at<double>(0, 3);
    extrinsic_array[i][4] = extrinsics[i].at<double>(1, 3);
    extrinsic_array[i][5] = extrinsics[i].at<double>(2, 3);
  }

  delete[] parameters;
  for(int i = 0; i < M; ++i)
  {
    delete[] extrinsic_array[i];
  }
  delete[] extrinsic_array;
}
