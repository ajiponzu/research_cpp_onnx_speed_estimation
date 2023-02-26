#include "../Functions.h"
using namespace Func;

#include <gdal_priv.h>

cv::Mat GeoCvt::get_multicolor_mat(const std::string& path)
{
	/* ラスタデータドライバの作成とファイル読み込み */
	GDALDataset* poDataset;
	GDALAllRegister();
	poDataset = (GDALDataset*)GDALOpen(path.c_str(), GDALAccess::GA_ReadOnly);
	if (poDataset == nullptr)
	{
		std::cout << "not found" << std::endl;
		return cv::Mat();
	}
	/* end */

	/* 緯度経度情報の取得 */
	double adfGeoTransform[6]{};
	if ((void*)(poDataset->GetProjectionRef()) != nullptr)
		printf("Projection is `%s'\n", poDataset->GetProjectionRef());
	if (poDataset->GetGeoTransform(adfGeoTransform) == CPLErr::CE_None)
	{
		printf("Origin = (%.6f,%.6f)\n",
			adfGeoTransform[0], adfGeoTransform[3]);
		printf("Pixel Size = (%.6f,%.6f)\n",
			adfGeoTransform[1], adfGeoTransform[5]);
	}
	/* end */

	/* ラスタデータの各バンドの読み込み */
	int nXSize = poDataset->GetRasterXSize();
	int nYSize = poDataset->GetRasterYSize();

	std::vector<cv::Mat> bands;
	for (auto&& poBand : poDataset->GetBands())
	{
		auto band_mat = cv::Mat(nYSize, nXSize, CV_8U);
		auto p_band_mat = band_mat.ptr<uint8_t>(0);
		uint8_t* pafScanline;
		pafScanline = (uint8_t*)CPLMalloc(sizeof(uint8_t) * nXSize * nYSize);
		poBand->RasterIO(GDALRWFlag::GF_Read, 0, 0, nXSize, nYSize,
			pafScanline, nXSize, nYSize, GDALDataType::GDT_Byte,
			0, 0);
		for (int i = 0; i < band_mat.total(); i++)
			p_band_mat[i] = pafScanline[i];

		bands.push_back(band_mat);
		CPLFree(pafScanline);
	}

	/* バンドをチャンネルに読み替えて統合する */
	cv::Mat img{};
	cv::merge(bands, img);

	if (img.channels() == 1)
		cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
	else
		cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
	/* end */
	/* end */

	return img;
}

cv::Mat GeoCvt::get_dsm_mat(const std::string& path)
{
	/* ラスタデータドライバの作成とファイル読み込み */
	GDALDataset* poDataset;
	GDALAllRegister();
	poDataset = (GDALDataset*)GDALOpen(path.c_str(), GDALAccess::GA_ReadOnly);
	if (poDataset == nullptr)
	{
		std::cout << "not found" << std::endl;
		return cv::Mat();
	}
	/* end */

	/* 緯度経度情報の取得 */
	double adfGeoTransform[6]{};
	if ((void*)(poDataset->GetProjectionRef()) != nullptr)
		printf("Projection is `%s'\n", poDataset->GetProjectionRef());
	if (poDataset->GetGeoTransform(adfGeoTransform) == CPLErr::CE_None)
	{
		printf("Origin = (%.6f,%.6f)\n",
			adfGeoTransform[0], adfGeoTransform[3]);
		printf("Pixel Size = (%.6f,%.6f)\n",
			adfGeoTransform[1], adfGeoTransform[5]);
	}
	/* end */

	/* ラスタデータの各バンドの読み込み */
	int nXSize = poDataset->GetRasterXSize();
	int nYSize = poDataset->GetRasterYSize();

	std::vector<cv::Mat> bands;
	for (auto&& poBand : poDataset->GetBands())
	{
		auto band_mat = cv::Mat(nYSize, nXSize, CV_32F);
		auto p_band_mat = band_mat.ptr<float_t>(0);
		float_t* pafScanline;
		pafScanline = (float_t*)CPLMalloc(sizeof(float_t) * nXSize * nYSize);
		poBand->RasterIO(GDALRWFlag::GF_Read, 0, 0, nXSize, nYSize,
			pafScanline, nXSize, nYSize, GDALDataType::GDT_Float32,
			0, 0);
		for (int i = 0; i < band_mat.total(); i++)
			p_band_mat[i] = pafScanline[i];

		bands.push_back(band_mat);
		CPLFree(pafScanline);
	}

	/* バンドをチャンネルに読み替えて統合する */
	cv::Mat img{};
	cv::merge(bands, img);
	/* end */
	/* end */

	return img;
}