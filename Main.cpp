#include "Utility.h"
#include "CarDetector.h"

constexpr std::string g_video_code = "hiru";
//constexpr std::string g_video_code = "yugata";

constexpr auto g_ortho_code = "ortho";
constexpr auto g_road_num = 4;
constexpr auto g_proc_imgsz = 320;

int main()
{
	GuiHandler::Initialize();
	ResourceProvider::Init(g_road_num, g_video_code, g_ortho_code);
	CarDetector carDetector(std::format(L"resources/dnns/yolov5x-{}.onnx", g_proc_imgsz),
		//CarDetector carDetector(std::format(L"resources/dnns/yolov3-tiny-{}.onnx", g_proc_imgsz),
		cv::Size(g_proc_imgsz, g_proc_imgsz));

	GuiHandler::SetVideoResource(std::format("resources/{}/input.mp4", g_video_code));
	GuiHandler::SetRenderer(carDetector.CreateRenderer());

	while (GuiHandler::EventPoll())
	{
		if (GuiHandler::MouseClickedL())
		{
			const auto& [x, y] = GuiHandler::GetClickPoint();
			cv::Point tl(std::max(0, x - g_proc_imgsz / 2), std::max(0, y - g_proc_imgsz / 2));
			cv::Point br(std::min(1920, x + g_proc_imgsz / 2), std::min(1080, y + g_proc_imgsz / 2));
			carDetector.SetRect(cv::Rect(tl, br));
		}

		if (GuiHandler::MouseClickedL() || GuiHandler::IsRunning())
		{
			const cv::Mat frame = GuiHandler::GetFrame();
			carDetector.Run(frame);
		}

		GuiHandler::Render();
	}
	cv::destroyAllWindows();
	std::cout << "end....." << std::endl;

	return 0;
}