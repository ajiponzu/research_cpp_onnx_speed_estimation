#include "../Detector.h"

DnnDetector::DnnDetector(const std::wstring& model_path, const cv::Size& proc_imgsz) : m_imgsz(proc_imgsz)
{
	m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
	m_sessionOptions = Ort::SessionOptions();

	const auto& available_providers = Ort::GetAvailableProviders();
	const auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
	OrtCUDAProviderOptions cuda_option;
	if (cuda_available == available_providers.end())
	{
		std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
		std::cout << "Inference device: CPU" << std::endl;
	}
	else
	{
		std::cout << "Inference device: GPU" << std::endl;
		m_sessionOptions.AppendExecutionProvider_CUDA(cuda_option);
	}
	m_session = Ort::Session(m_env, model_path.c_str(), m_sessionOptions);

	const auto& input_type_info = m_session.GetInputTypeInfo(0);
	const auto& input_tensor_shape = input_type_info.GetTensorTypeAndShapeInfo().GetShape();
	m_isDynamicInputShape = false;
	// checking if width and height are dynamic
	if (input_tensor_shape[2] == -1 && input_tensor_shape[3] == -1)
	{
		std::cout << "Dynamic input shape" << std::endl;
		m_isDynamicInputShape = true;
	}

	for (const auto& shape : input_tensor_shape)
	{
		std::cout << "Input shape: " << shape << std::endl;
		m_inputShape.emplace_back(shape);
	}

	Ort::AllocatorWithDefaultOptions allocator;
	m_inputName = std::string(m_session.GetInputNameAllocated(0, allocator).get());
	m_outputName = std::string(m_session.GetOutputNameAllocated(0, allocator).get());
	std::cout << "Input name: " << m_inputName << std::endl;
	std::cout << "Output name: " << m_outputName << std::endl;
}

std::vector<float> DnnDetector::LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size)
{
	auto in_h = (float)(src.rows);
	auto in_w = (float)(src.cols);
	auto out_h = (float)(out_size.height);
	auto out_w = (float)(out_size.width);

	float scale = std::min(out_w / in_w, out_h / in_h);

	auto mid_h = (int)(in_h * scale);
	auto mid_w = (int)(in_w * scale);

	cv::resize(src, dst, cv::Size(mid_w, mid_h));

	auto top = ((int)out_h - mid_h) / 2;
	auto down = ((int)out_h - mid_h + 1) / 2;
	auto left = ((int)out_w - mid_w) / 2;
	auto right = ((int)out_w - mid_w + 1) / 2;

	cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

	std::vector<float> pad_info{ (float)left, (float)top, scale };
	return pad_info;
}