#include "../CarDetector.h"

static constexpr auto g_kilo_ratio = 0.001f;
static constexpr auto g_font_face = cv::FONT_HERSHEY_DUPLEX;
static constexpr auto g_font_scale = 1.0;
static constexpr auto g_thickness = 1;

static cv::Point2f get_trans_point(const cv::Point& point)
{
	const auto& color_mask_list = ResourceProvider::GetRoadColorMaskList();
	auto& json_hash_list = ResourceProvider::GetJsonHashList();

	for (size_t i = 0; i < color_mask_list.size(); i++)
	{
		auto& json_hash = json_hash_list[i];
		const auto& color_mask = color_mask_list[i];

		const auto& pix = color_mask.at<cv::Vec3b>(point);
		const auto color_str = std::format("{},{},{}", pix[0], pix[1], pix[2]);
		if (json_hash.find(color_str) == json_hash.end())
			continue;

		const cv::Mat& trans_mat = json_hash[color_str];
		std::vector<cv::Point2f> src_corners{ static_cast<cv::Point2f>(point) };
		decltype(src_corners) dst_corners{};
		cv::perspectiveTransform(src_corners, dst_corners, trans_mat);

		return dst_corners[0];
	}

	return cv::Point2f(-1.0f, -1.0f);
}

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

static double calc_speed(const double& sum_delta, const uint64_t& time, const double& fps)
{
	auto speed_per_second = (sum_delta / time) * fps;
	return speed_per_second * 3600;
}

void CarDetector::DetectedCar::CalcSpeed(const uint64_t& tracking_time, const double& magni)
{
	const auto distance = calc_distance(m_startPointTransed, m_curPointTransed, magni) * g_kilo_ratio;
	m_speed = calc_speed(distance, tracking_time, GuiHandler::GetFPS());
}

void CarDetector::DetectedCar::Init()
{
	const auto car_center = Func::Img::calc_rect_center(m_shape);
	m_startPointTransed = get_trans_point(car_center);
	m_startTime = m_curTime = GuiHandler::GetFrameCount();
}

void CarDetector::DetectedCar::DrawOnImage(cv::Mat& img, const std::vector<std::string>& class_names) const
{
	//const auto class_str = std::format("{} {:.2f}", class_names[m_carType], m_detectedScore);
	const auto class_str = std::format("{} {:.1f}", class_names[m_carType], m_speed);

	int base_line = 0;
	const auto font_size = cv::getTextSize(class_str, g_font_face, g_font_scale, g_thickness, &base_line);

	cv::rectangle(img, m_shape, cv::Scalar(0, 0, 255), 2);
	cv::rectangle(img,
		cv::Point(m_shape.tl().x, m_shape.tl().y - font_size.height - 5),
		cv::Point(m_shape.tl().x + font_size.width, m_shape.tl().y),
		cv::Scalar(0, 0, 255), -1);
	cv::putText(img, class_str,
		cv::Point(m_shape.tl().x, m_shape.tl().y - 5),
		g_font_face, g_font_scale, cv::Scalar(255, 255, 255), g_thickness);
}

void CarDetector::DetectedCar::DrawOnOrtho(cv::Mat& ortho) const
{
	cv::circle(ortho, static_cast<cv::Point>(m_startPointTransed), 5, cv::Scalar(0, 255, 0), -1);
	cv::circle(ortho, static_cast<cv::Point>(m_curPointTransed), 5, cv::Scalar(0, 0, 255), -1);
}

bool CarDetector::DetectedCar::TryTracking(const DetectionResult& detection)
{
	const auto car_center = Func::Img::calc_rect_center(m_shape);
	const auto box_center = Func::Img::calc_rect_center(detection.box);

	if (cv::norm(car_center - box_center) > 40)
		return false;

	*this = detection;
	m_curPointTransed = get_trans_point(box_center);

	m_curTime = GuiHandler::GetFrameCount();
	CalcSpeed(m_curTime - m_startTime, 0.2);

	return true;
}