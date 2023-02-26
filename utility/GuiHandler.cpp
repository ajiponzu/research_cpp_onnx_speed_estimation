#include "../Utility.h"
#include <Windows.h>

/* GuiHandlerのstatic変数初期化 */
uint64_t GuiHandler::s_frameCount = 0;
bool GuiHandler::s_isRunning = true;
bool GuiHandler::s_wndUpdate = true;
bool GuiHandler::s_useVideo = false;
bool GuiHandler::s_mouseClickedL = false;
std::pair<int, int> GuiHandler::s_clickPoint;
cv::Mat GuiHandler::s_displayImg{};
cv::Mat GuiHandler::s_currentFrame{};
cv::VideoCapture GuiHandler::s_videoCapture;
std::unique_ptr<Renderer> GuiHandler::s_ptrRenderer = nullptr;
std::unordered_set<int>  GuiHandler::s_keyEventTable{};
/* end */

/* global変数 */
static constexpr auto gWndName = "display";
static constexpr auto gFrameInterval = 1;
static constexpr auto base_path = "outputs/screenshots";
/* end */

void GuiHandler::ScreenShot()
{
	std::string path = std::format("{}/result_{}.png", base_path, s_frameCount);
	cv::imwrite(path, s_displayImg);
	path = std::format("{}/frame_{}.png", base_path, s_frameCount);
	cv::imwrite(path, s_currentFrame);
}

void GuiHandler::RecvMouseMsg(int event, int x, int y, int flag, void* callBack)
{
	switch (event) {
	case cv::EVENT_LBUTTONDOWN: // マウス左クリック
		s_clickPoint = std::make_pair(x, y);
		s_mouseClickedL = true;
		break;
	case cv::EVENT_RBUTTONDOWN: // マウス右クリック
		s_isRunning = !s_isRunning;
		break;
	default:
		break;
	}
}

void GuiHandler::HandleInputKey(const int& key)
{
	switch (key)
	{
	case 27: // ESCキー
		s_wndUpdate = false;
		s_keyEventTable.insert(27);
		break;
	case ' ': // Spaceキー
		s_keyEventTable.insert((int)(' '));
		break;
	default:
		break;
	}
}

void GuiHandler::ClearEventFlags()
{
	s_keyEventTable.clear();
}

void GuiHandler::Initialize()
{
	// FreeConsole(); // 本番環境でのコンソール非表示関数

	cv::namedWindow(gWndName, cv::WindowFlags::WINDOW_FULLSCREEN);
	cv::setMouseCallback(gWndName, RecvMouseMsg);
}

bool GuiHandler::EventPoll()
{
	s_mouseClickedL = false;
	HandleInputKey(cv::waitKey(gFrameInterval));

	if (s_useVideo && s_isRunning) // ビデオ使用時かつ再生中
	{
		s_videoCapture.read(s_currentFrame);
		if (s_currentFrame.empty())
		{
			s_wndUpdate = false;
			return s_wndUpdate;
		}
		s_frameCount++;
	}
	s_currentFrame.copyTo(s_displayImg);

	if (!s_videoCapture.isOpened() && s_useVideo)
		s_wndUpdate = false;

	return s_wndUpdate;
}

void GuiHandler::Render()
{
	if (s_ptrRenderer)
		s_ptrRenderer->Render(s_displayImg);

	if (GetKeyEvent((int)(' ')))
		ScreenShot();

	cv::imshow(gWndName, s_displayImg);
	ClearEventFlags();
}

void GuiHandler::SetVideoResource(const std::string& path)
{
	if (s_videoCapture.open(path))
	{
		const int wid = (int)s_videoCapture.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
		const int high = (int)s_videoCapture.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);
		s_currentFrame = cv::Mat(high, wid, CV_8UC3);
		s_displayImg = cv::Mat(high, wid, CV_8UC3);

		s_useVideo = true;
	}
}

void GuiHandler::SetRenderer(Renderer* ptrRenderer)
{
	s_ptrRenderer.reset(ptrRenderer);
}

double GuiHandler::GetFPS()
{
	if (!s_useVideo)
		return 0.0;

	return s_videoCapture.get(cv::CAP_PROP_FPS);
}

const std::pair<int, int>& GuiHandler::GetClickPoint()
{
	std::cout << std::format("click: ({}, {})", s_clickPoint.first, s_clickPoint.second) << std::endl;
	return s_clickPoint;
}