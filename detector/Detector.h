#pragma once
#include "../Utility.h"

struct DetectionResult
{
	cv::Rect box{};
	float score = 0.0f;
	int class_idx = 0;
};

class Detector
{
private:
	friend class CarDetector;
protected:
	virtual std::vector<DetectionResult>
		Run(const cv::Mat& img) = 0;
};

class DnnDetector : public Detector
{
protected:
	cv::Size m_imgsz;
	Ort::Env m_env{ nullptr };
	Ort::SessionOptions m_sessionOptions{ nullptr };
	Ort::Session m_session{ nullptr };
	std::string m_inputName;
	std::string m_outputName;
	std::vector<int64_t> m_inputShape{};
	bool m_isDynamicInputShape;

	virtual std::vector<DetectionResult>
		Run(const cv::Mat& img) override = 0;

	static std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size);

public:
	DnnDetector(const std::wstring& model_path, const cv::Size& proc_imgsz);
};

class YoloDetector : public DnnDetector {
public:
	YoloDetector(const std::wstring& model_path,
		const cv::Size& imgsz,
		const float& conf_threshold = 0.4,
		const float& iou_threshold = 0.6)
		: DnnDetector(model_path, imgsz)
		, m_confThr(conf_threshold)
		, m_iouThr(iou_threshold)
	{}

	virtual std::vector<DetectionResult>
		Run(const cv::Mat& img) override;

private:
	float m_confThr;
	float m_iouThr;
};
