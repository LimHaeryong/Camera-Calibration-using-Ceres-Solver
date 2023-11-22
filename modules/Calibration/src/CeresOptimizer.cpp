#include "Calibration/CeresOptimizer.h"

void CeresOptimizer::solve()
{
	ceres::Solver::Summary summary;
	ceres::Solve(options, problem.get(), &summary);
}

void CeresOptimizer::add_residual(const cv::Point2f &image_point, const cv::Point3f &object_point, std::vector<double> &intrinsic, std::vector<double> &extrinsic)
{
	switch (optimizer_type)
	{
	case Type::distortion_geometric:
		ceres::CostFunction *cost_function = reprojection_error::create(image_point, object_point);
		problem->AddResidualBlock(cost_function, NULL, intrinsic.data(), extrinsic.data());
		break;
	}
}

void CeresOptimizer::add_residual(const cv::Point2f &src_point, const cv::Point2f &dst_point, std::vector<double> &homography)
{
	switch (optimizer_type)
	{
		case Type::homography_geometric:
			ceres::CostFunction *cost_function = homography_reproj_error::create(src_point, dst_point);
			problem->AddResidualBlock(cost_function, NULL, homography.data());
			break;
	}
}