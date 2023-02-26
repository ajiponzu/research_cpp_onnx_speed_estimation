#pragma once
#include "detector/Detector.h"
#include "Utility.h"

class CarDetector
{
private:
	class SpeedIndicator
	{
	public:
		static double CalcSpeed(const double& sum_delta, const uint64_t& time, const double& fps);
		static double CalcDelta(const cv::Point2f& old_point, const cv::Point2f& new_point, const double& magni);
	};

	class ThisRenderer : public Renderer
	{
	private:
		std::shared_ptr<CarDetector> m_ptrDetector;

		void Render(cv::Mat& img) override;

		void DrawDetections(cv::Mat& img);
	public:
		ThisRenderer(CarDetector* ptr) : m_ptrDetector(ptr) {}
	};

	bool m_resetDetect = false;
	uint64_t m_trackingCount = 0;

	cv::Point2f m_startPoint;
	cv::Rect m_detectArea;
	cv::Point2f m_curPoint;

	std::unique_ptr<Detector> m_ptrDetector;
	std::vector<std::string> m_classNames;
	std::vector<DetectionResult> m_detections;

public:
	CarDetector(const std::wstring& model_path = L"", const cv::Size& proc_imgsz = cv::Size(640, 640));

	ThisRenderer* CreateRenderer() { return new ThisRenderer(this); }

	void Run(const cv::Mat& img);
	void SetRect(const cv::Rect& rect);
};
