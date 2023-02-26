#include "../Detector.h"

static void ReshapeImageToTensor(const cv::Mat& src, cv::Mat& dst)
{
	cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);  // BGR -> RGB
	src.convertTo(dst, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255

	dst = dst.reshape(1, (int)(src.total()));
	cv::transpose(dst, dst);
}

static std::unordered_set<int> g_class_set;
static void GetBestClassInfo(std::vector<float>::iterator& itr, const int& num_classes,
	float& best_conf, int& best_class_id)
{
	// first 5 element are box and obj confidence
	// float[] -> { ?, ?, ?, ?, ?, class_0_conf, class_1_conf, ..... }
	best_class_id = 5;
	best_conf = 0;

	g_class_set.insert(2 + best_class_id); // car
	g_class_set.insert(5 + best_class_id); // bus
	g_class_set.insert(7 + best_class_id); // truck

	for (int i = 5; i < num_classes + 5; i++)
	{
		if (g_class_set.find(i) == g_class_set.end())
			continue;

		if (itr[i] > best_conf)
		{
			best_conf = itr[i];
			best_class_id = i - 5;
		}
	}
}

static float clip(const float& n, const float& lower, const float& upper)
{
	return std::max(lower, std::min(n, upper));
}

static void ScaleCoordinates(std::vector<DetectionResult>& detections, float pad_w, float pad_h,
	float scale, const cv::Size& img_shape)
{
	for (auto& detection : detections)
	{
		float x1 = (detection.box.tl().x - pad_w) / scale;  // x padding
		float y1 = (detection.box.tl().y - pad_h) / scale;  // y padding
		float x2 = (detection.box.br().x - pad_w) / scale;  // x padding
		float y2 = (detection.box.br().y - pad_h) / scale;  // y padding

		x1 = clip(x1, 0.0f, (float)(img_shape.width));
		y1 = clip(y1, 0.0f, (float)(img_shape.height));
		x2 = clip(x2, 0.0f, (float)(img_shape.width));
		y2 = clip(y2, 0.0f, (float)(img_shape.height));

		detection.box = cv::Rect(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
	}
}

static std::vector<DetectionResult> PostProcessing(const std::vector<Ort::Value>& outputs,
	float pad_w, float pad_h, float scale, const cv::Size& img_shape,
	float conf_thres, float iou_thres)
{
	std::vector<cv::Rect> boxes;
	std::vector<float> confs;
	std::vector<int> class_ids;
	std::vector<DetectionResult> results;

	constexpr int item_attr_size = 5;
	const auto& output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	const auto& batch_size = (int)output_shape[0];
	const auto* raw_output = outputs[0].GetTensorData<float>();
	const auto& count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	std::vector<float> output(raw_output, raw_output + count);

	// number of classes, e.g. 80 for coco dataset
	const auto num_classes = (int)output_shape[2] - item_attr_size;
	const auto elements_in_batch = (int)(output_shape[1] * output_shape[2]);

	// iterating all images in the batch
	for (auto itr = output.begin(); itr != output.begin() + elements_in_batch; itr += output_shape[2])
	{
		const auto& cls_conf = itr[4];

		if (cls_conf > conf_thres)
		{
			int center_x = (int)(itr[0]);
			int center_y = (int)(itr[1]);
			int width = (int)(itr[2]);
			int height = (int)(itr[3]);
			int left = center_x - width / 2;
			int top = center_y - height / 2;

			float obj_conf;
			int class_id;
			GetBestClassInfo(itr, num_classes, obj_conf, class_id);

			float confidence = cls_conf * obj_conf;

			boxes.emplace_back(left, top, width, height);
			confs.emplace_back(confidence);
			class_ids.emplace_back(class_id);
		}
	}

	// run NMS
	std::vector<int> nms_indices;
	cv::dnn::NMSBoxes(boxes, confs, conf_thres, iou_thres, nms_indices);

	for (int idx : nms_indices)
	{
		DetectionResult result;
		result.box = cv::Rect(boxes[idx]);

		result.score = confs[idx];
		result.class_idx = class_ids[idx];
		results.emplace_back(result);
	}
	ScaleCoordinates(results, pad_w, pad_h, scale, img_shape);

	return results;
}

std::vector<DetectionResult>
YoloDetector::Run(const cv::Mat& img)
{
	cv::Mat input_img = img.clone();

	std::vector<float> pad_info = LetterboxImage(input_img, input_img, m_imgsz);
	const float pad_w = pad_info[0];
	const float pad_h = pad_info[1];
	const float scale = pad_info[2];

	cv::Mat tensor{};
	ReshapeImageToTensor(input_img, tensor);

	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
		OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

	std::vector<Ort::Value> inputs;
	inputs.push_back(Ort::Value::CreateTensor<float>(
		memory_info, tensor.ptr<float>(), tensor.total(),
		m_inputShape.data(), m_inputShape.size()
		));

	// inference
	const auto input_name = m_inputName.c_str();
	const auto output_name = m_outputName.c_str();
	auto outputs = m_session.Run(Ort::RunOptions{ nullptr }, &input_name, inputs.data(), inputs.size(), &output_name, 1);

	/*** Post-process ***/
	// result: n * 7
	// batch index(0), top-left x/y (1,2), bottom-right x/y (3,4), score(5), class id(6)
	auto result = PostProcessing(outputs, pad_w, pad_h, scale, img.size(), m_confThr, m_iouThr);

	return result;
}
