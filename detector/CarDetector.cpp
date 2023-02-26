#include "../CarDetector.h"
#include "Detector.h"

CarDetector::CarDetector(const std::wstring& model_path, const cv::Size& proc_imgsz)
{
	m_ptrDetector.reset(new YoloDetector(model_path, proc_imgsz, 0.2f, 0.2f));
	
	/* 初期実行が遅いため，一度動かしておく */
	const auto black_img = cv::Mat::zeros(proc_imgsz, CV_8UC3);
	m_ptrDetector->Run(black_img);
	/* end */
}

void CarDetector::Run(const cv::Mat& img)
{
	if (m_detectArea == cv::Rect())
		return;

	m_detections = m_ptrDetector->Run(img(m_detectArea));
	m_resetDetect = false;
	m_trackingCount++;

	for (auto& detection : m_detections)
	{
		detection.box.x += m_detectArea.x;
		detection.box.y += m_detectArea.y;
	}
}

void CarDetector::ThisRenderer::Render(cv::Mat& img)
{
	const auto& road_mask = ResourceProvider::GetRoadMask();
	cv::addWeighted(img, 0.85, road_mask, 0.15, 1.0, img);

	DrawDetections(img);
	cv::rectangle(img, m_ptrDetector->m_detectArea, cv::Scalar(0, 0, 255), 2);
}

void CarDetector::ThisRenderer::DrawDetections(cv::Mat& img)
{
	const auto& detections = m_ptrDetector->m_detections;

	if (detections.empty())
		return;

	cv::circle(img, m_ptrDetector->m_startPoint, 3, cv::Scalar(0, 255, 0), -1);
	cv::circle(img, m_ptrDetector->m_curPoint, 3, cv::Scalar(0, 0, 255), -1);

	for (const auto& detection : detections)
	{
		cv::Mat road_mask;
		cv::cvtColor(ResourceProvider::GetRoadMask(), road_mask, cv::COLOR_BGR2GRAY);
		const auto& center = Func::Img::calc_rect_center(detection.box);
		const auto& pix = road_mask.at<uint8_t>(center);

		if (pix == 0)
			continue;

		cv::rectangle(img, detection.box, cv::Scalar(255, 0, 0), 2);
	}
}

void CarDetector::SetRect(const cv::Rect& rect)
{
	m_detectArea = rect;
	m_resetDetect = true;
	m_trackingCount = 0;
	m_startPoint = m_curPoint = cv::Point2f();
}