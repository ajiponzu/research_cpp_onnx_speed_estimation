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
		uint64_t m_startTime = 0;
		uint64_t m_curTime = 0;
		double m_speed = 0.0;
		cv::Rect m_shape;
		cv::Point2f m_startPointTransed;
		cv::Point2f m_curPointTransed;

		DetectedCar(const int& car_type, const float& detected_score, const cv::Rect& shape)
			: m_carType(car_type)
			, m_detectedScore(detected_score)
			, m_shape(shape)
		{
			Init();
		}

		void CalcSpeed(const uint64_t& tracking_time, const double& magni);

		void Init();

	public:
		DetectedCar(const DetectionResult& detection)
		{
			*this = DetectedCar(detection.class_idx, detection.score, detection.box);
		}

		DetectedCar operator = (const DetectionResult& detection)
		{
			m_carType = detection.class_idx;
			m_detectedScore = detection.score;
			m_shape = detection.box;

			return *this;
		}

		const double& GetSpeed() const { return m_speed; }
		const bool IsUntracked() const { return m_curTime != GuiHandler::GetFrameCount(); }

		void DrawOnImage(cv::Mat& img, const std::vector<std::string>& class_names) const;
		void DrawOnOrtho(cv::Mat& ortho) const;
		bool TryTracking(const DetectionResult& detection);
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
	cv::Rect m_detectAreaUpdated;
	std::unique_ptr<Detector> m_ptrDetector;
	std::vector<DetectedCar> m_detectedCars;

public:
	CarDetector(const std::wstring& model_path = L"", const cv::Size& proc_imgsz = cv::Size(640, 640));

	ThisRenderer* CreateRenderer() { return new ThisRenderer(this); }

	void Run(const cv::Mat& img);
	void SetRect(const cv::Rect& rect);
};
