#include "../Utility.h"

cv::Mat ResourceProvider::s_roadMask;
cv::Mat ResourceProvider::s_orthoDsm;
cv::Mat ResourceProvider::s_orthoTif;
cv::Mat ResourceProvider::s_orthoRoadMask;
cv::Mat ResourceProvider::s_roadColorMask;
std::vector<cv::Mat> ResourceProvider::s_roadColorMaskList;
JsonHashList ResourceProvider::s_jsonHashList;

void ResourceProvider::Init(const int& road_num, const std::string& video_code, const std::string& ortho_code)
{
	const auto road_mask_path = std::format("resources/{}/road_mask.png", video_code);
	const auto ortho_dem_path = std::format("resources/{}/dsm.tif", ortho_code);
	const auto ortho_tif_path = std::format("resources/{}/ortho.tif", ortho_code);
	const auto ortho_road_mask_path = std::format("resources/{}/road_mask.png", ortho_code);
	const auto color_mask_path_base = std::format("resources/{}/road_color_mask", video_code);
	const auto json_path_base = std::format("resources/{}/hmg_area", video_code);

	s_roadMask = cv::imread(road_mask_path);
	s_orthoDsm = Func::GeoCvt::get_dsm_mat(ortho_dem_path);
	s_orthoTif = Func::GeoCvt::get_multicolor_mat(ortho_tif_path);
	s_orthoRoadMask = cv::imread(ortho_road_mask_path);

	for (int i = 0; i < road_num; i++)
	{
		const auto color_mask_path = std::format("{}{}.png", color_mask_path_base, i);
		const auto json_path = std::format("{}{}.json", json_path_base, i);

		s_roadColorMaskList.push_back(cv::imread(color_mask_path));
		s_jsonHashList.push_back(Func::Json::get_hmg_from_json(json_path));
	}

	s_roadColorMask = s_roadColorMaskList[0] + s_roadColorMaskList[1] + s_roadColorMaskList[2] + s_roadColorMaskList[3];
}