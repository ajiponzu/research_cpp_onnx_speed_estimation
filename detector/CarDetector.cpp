#include "../CarDetector.h"
#include "Detector.h"

static std::vector<std::string> g_class_names{};

static std::vector<std::string> load_names(const std::string& path)
{
	// load class names
	std::vector<std::string> class_names;
	std::ifstream infile(path);
	if (infile.is_open())
	{
		std::string line;
		while (getline(infile, line))
			class_names.emplace_back(line);

		infile.close();
	}
	else
		std::cerr << "Error loading the class names!" << std::endl;

	return class_names;
}

CarDetector::CarDetector(const std::wstring& model_path, const cv::Size& proc_imgsz)
{
	m_ptrDetector.reset(new YoloDetector(model_path, proc_imgsz, 0.2f, 0.2f));
	g_class_names = load_names("resources/dnns/classes/coco.names");
	
	/* DNNモデルは初期実行が遅いため，一度動かしておく */
	const auto black_img = cv::Mat::zeros(proc_imgsz, CV_8UC3);
	m_ptrDetector->Run(black_img);
	/* end */
}

void CarDetector::Run(const cv::Mat& img)
{
	if (m_detectArea == cv::Rect())
		return;

	auto detections = m_ptrDetector->Run(img(m_detectArea));

	m_detectedCars.clear();
	cv::Mat road_mask;
	cv::cvtColor(ResourceProvider::GetRoadMask(), road_mask, cv::COLOR_BGR2GRAY);

	for (auto& detection : detections)
	{
		auto& box = detection.box;
		box.x += m_detectArea.x;
		box.y += m_detectArea.y;

		const auto box_center = Func::Img::calc_rect_center(box);
		if (!Func::Img::is_on_mask(road_mask, box_center))
			continue;

		m_detectedCars.push_back(DetectedCar(detection));
	}
}

void CarDetector::ThisRenderer::Render(cv::Mat& img)
{
	const auto& road_mask = ResourceProvider::GetRoadMask();
	cv::addWeighted(img, 0.85, road_mask, 0.15, 1.0, img);

	DrawDetections(img);
	cv::rectangle(img, m_ptrDetector->m_detectArea, cv::Scalar(255, 0, 0), 2);
}

void CarDetector::ThisRenderer::DrawDetections(cv::Mat& img)
{
	const auto& detected_cars = m_ptrDetector->m_detectedCars;

	if (detected_cars.empty())
		return;

	for (const auto& detected_car : detected_cars)
		detected_car.DrawOnImage(img, g_class_names);
}

void CarDetector::SetRect(const cv::Rect& rect)
{
	m_detectArea = rect;
	m_detectedCars.clear();
}