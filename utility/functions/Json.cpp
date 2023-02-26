#include "../Functions.h"
using namespace Func;

Json::JsonDataHash Json::get_hmg_from_json(const std::string& path)
{
	cv::FileStorage json(path, cv::FileStorage::READ);

	JsonDataHash result_hash;
	const auto points_num = (int)(json["points_num"]);

	for (int i = 0; i < points_num; i++)
	{
		const auto color_str_key = std::format("color_{}", i);
		const auto hmg_mat_key = std::format("hmg_mat_{}", i);
		const auto color_str = json[color_str_key].string();
		const auto hmg_mat = json[hmg_mat_key].mat();

		result_hash[color_str] = hmg_mat;
	}

	return result_hash;
}