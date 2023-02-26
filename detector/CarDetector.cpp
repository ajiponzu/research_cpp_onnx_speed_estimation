#include "../CarDetector.h"
#include "Detector.h"

static std::vector<std::string> g_class_names{};
static cv::Mat g_road_mask_gray;

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
	cv::cvtColor(ResourceProvider::GetRoadMask(), g_road_mask_gray, cv::COLOR_BGR2GRAY);

	/* DNNモデルは初期実行が遅いため，一度動かしておく */
	const auto black_img = cv::Mat::zeros(proc_imgsz, CV_8UC3);
	m_ptrDetector->Run(black_img);
	/* end */
}

void CarDetector::Run(const cv::Mat& img)
{
	if (m_detectArea == cv::Rect())
		return;

	m_detectArea = m_detectAreaUpdated;

	auto detections = m_ptrDetector->Run(img(m_detectArea));

	const auto itr_cars_begin = m_detectedCars.begin();
	const auto itr_cars_end = m_detectedCars.end();
	for (auto& detection : detections)
	{
		auto& box = detection.box;
		box.x += m_detectArea.x;
		box.y += m_detectArea.y;

		if (box.area() > 3600)
			continue;

		const auto box_center = Func::Img::calc_rect_center(box);
		if (!Func::Img::is_on_mask(g_road_mask_gray, box_center))
			continue;

		auto itr = itr_cars_begin;
		for (; itr != itr_cars_end; itr++)
		{
			if (itr->TryTracking(detection))
				break;
		}

		if (itr == itr_cars_end)
			m_detectedCars.push_back(DetectedCar(detection));
	}

	for (auto itr = m_detectedCars.begin(); itr != m_detectedCars.end();)
	{
		if (itr->IsUntracked())
			m_detectedCars.erase(itr);
		else
			itr++;
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

	auto ortho = ResourceProvider::GetOrthoTif().clone();
	const auto& ortho_road_mask = ResourceProvider::GetOrthoRoadMask();
	cv::addWeighted(ortho, 0.85, ortho_road_mask, 0.15, 1.0, ortho);

	uint64_t car_id = 0;
	for (const auto& detected_car : detected_cars)
	{
		detected_car.DrawOnImage(img, g_class_names);
		detected_car.DrawOnOrtho(ortho);
		std::cout << std::format("car_{}: {:.1f} [km/h]", car_id, detected_car.GetSpeed()) << std::endl;
		car_id++;
	}
	cv::resize(ortho, ortho, cv::Size(), 0.5, 0.5);
	cv::imshow("ortho", ortho);
}

void CarDetector::SetRect(const cv::Rect& rect)
{
	m_detectArea = m_detectAreaUpdated = rect;
	m_detectedCars.clear();
}