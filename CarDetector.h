#pragma once
#include "detector/Detector.h"
#include "Utility.h"

class CarDetector
{
private:
	class DetectedCar
	{
	private:
		int m_carType = 2;
		float m_detectedScore = 0.0f;
		size_t m_trackingCount = 0;
		double m_speed = 0.0;
		cv::Point m_startPoint;
		cv::Rect m_shape;

	public:
		DetectedCar(const int& car_type, const float& detected_score, const cv::Rect& shape)
			: m_carType(car_type)
			, m_detectedScore(detected_score)
			, m_shape(shape) {}

		DetectedCar(const DetectionResult& detection)
		{
			*this = DetectedCar(detection.class_idx, detection.score, detection.box);
		}

		void DrawOnImage(cv::Mat& img, const std::vector<std::string>& class_names) const;

		//static double CalcSpeed(const double& sum_delta, const uint64_t& time, const double& fps);
		//static double CalcDelta(const cv::Point2f& old_point, const cv::Point2f& new_point, const double& magni);
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

	cv::Rect m_detectArea;
	std::unique_ptr<Detector> m_ptrDetector;
	std::vector<DetectedCar> m_detectedCars;

public:
	CarDetector(const std::wstring& model_path = L"", const cv::Size& proc_imgsz = cv::Size(640, 640));

	ThisRenderer* CreateRenderer() { return new ThisRenderer(this); }

	void Run(const cv::Mat& img);
	void SetRect(const cv::Rect& rect);
};
