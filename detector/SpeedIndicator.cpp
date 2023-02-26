#include "../CarDetector.h"

static constexpr auto g_kilo_ratio = 0.001f;

static double calc_distance(const cv::Point2f& old_point, const cv::Point2f& new_point, const double& magni)
{
	const auto& dsm = ResourceProvider::GetOrthoDsm();

	const auto old_z_meter = dsm.at<float_t>(old_point);
	const auto new_z_meter = dsm.at<float_t>(new_point);
	const auto delta_z_meter = new_z_meter - old_z_meter;

	const auto delta_xy = new_point - old_point;
	const auto delta_xy_meter = delta_xy * magni;
	const auto dist_vec_3d = cv::Point3f(delta_xy_meter.x, delta_xy_meter.y, delta_z_meter);

	return cv::norm(dist_vec_3d);
}

static cv::Point2f get_trans_point(const cv::Mat& trans, const cv::Point2f& point)
{
	std::vector<cv::Point2f> src_corners{};
	std::vector<cv::Point2f> dst_corners{};

	src_corners.push_back(point);
	cv::perspectiveTransform(src_corners, dst_corners, trans);

	return dst_corners[0];
}

double CarDetector::SpeedIndicator::CalcSpeed(const double& sum_delta, const uint64_t& time, const double& fps)
{
	auto speed_per_second = (sum_delta / time) * fps;
	return speed_per_second * 3600;
}

double CarDetector::SpeedIndicator::CalcDelta(const cv::Point2f& old_point, const cv::Point2f& new_point, const double& magni)
{
	const auto& color_mask_list = ResourceProvider::GetRoadColorMaskList();
	auto& json_hash_list = ResourceProvider::GetJsonHashList();

	for (size_t i = 0; i < color_mask_list.size(); i++)
	{
		auto& json_hash = json_hash_list[i];
		const auto& color_mask = color_mask_list[i];

		const auto& pix_old = color_mask.at<cv::Vec3b>(static_cast<cv::Point>(old_point));
		const auto color_str_old = std::format("{},{},{}", pix_old[0], pix_old[1], pix_old[2]);
		if (json_hash.find(color_str_old) == json_hash.end())
			continue;

		const auto& pix_new = color_mask.at<cv::Vec3b>(static_cast<cv::Point>(new_point));
		const auto color_str_new = std::format("{},{},{}", pix_new[0], pix_new[1], pix_new[2]);
		if (json_hash.find(color_str_new) == json_hash.end())
			continue;

		const cv::Mat& trans_mat_old = json_hash[color_str_old];
		const cv::Mat& trans_mat_new = json_hash[color_str_new];

		const auto transed_old = get_trans_point(trans_mat_old, old_point);
		const auto transed_new = get_trans_point(trans_mat_new, new_point);

		auto ortho = ResourceProvider::GetOrthoTif().clone();
		const auto& ortho_road_mask = ResourceProvider::GetOrthoRoadMask();
		cv::addWeighted(ortho, 0.85, ortho_road_mask, 0.15, 1.0, ortho);
		cv::circle(ortho, transed_old, 5, cv::Scalar(0, 255, 0), -1);
		cv::circle(ortho, transed_new, 5, cv::Scalar(0, 0, 255), -1);
		cv::resize(ortho, ortho, cv::Size(), 0.5, 0.5);
		cv::imshow("ortho", ortho);

		return calc_distance(transed_old, transed_new, magni) * g_kilo_ratio;
	}

	return 0.0;
}