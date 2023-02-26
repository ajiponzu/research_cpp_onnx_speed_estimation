#pragma once

namespace Func
{
	namespace Img
	{
		// get white-lane areas
		// color-image-segmentation by k-means
		cv::Mat get_white_lane(const cv::Mat& mat, const int& n_k = 3);

		// get shadow-area by lab statistic
		cv::Mat extract_shadow(const cv::Mat& mat);

		// apply with morphological method with shadow and get unshadow-area by bit_not
		cv::Mat get_unshadow(cv::Mat& shadow);

		// hmg_warp
		void warp_img_by_hmg(const cv::Mat& src, cv::Mat& dst, cv::Mat& hmg_layer,
			const std::vector<cv::Point2f>& src_pts, const std::vector<cv::Point2f>& dst_pts);

		cv::Mat binarize_img(cv::Mat& src, const int& channels = 1);
		cv::Mat binarize_block_img(cv::Mat& src, const int& channels = 1);
		cv::Mat contrast_local_area(cv::Mat& src);
		cv::Mat get_img_slice(const cv::Mat& src, const cv::Rect& area, const int& channels = 3);

		cv::Point calc_rect_center(const cv::Rect& rect);

		bool is_on_mask(const cv::Mat& gray_mask, const cv::Point& pt);
	};

	namespace GeoCvt
	{
		cv::Mat get_multicolor_mat(const std::string& path);

		cv::Mat get_dsm_mat(const std::string& path);
	};

	namespace Json
	{
		// key: color_str; ex) "255,255,255",  value: hmg_mat; cv::Mat
		using JsonDataHash = std::unordered_map<std::string, cv::Mat>;

		JsonDataHash get_hmg_from_json(const std::string& path);
	};
};
