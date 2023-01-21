#include <vector>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include <fstream>
#include <sstream>

struct NormalizedData
{
	cv::Mat T;
    std::vector<cv::Point2f> points;
};

NormalizedData normalizeData(const std::vector<cv::Point2f>& points)
{
	int ptsNum = points.size();

	// calculate means (they will be the center of coordinate systems)
	float meanx = 0.0, meany = 0.0;
	for (int i = 0; i < ptsNum; i++)
	{
		auto& p = points[i];
		meanx += p.x;
		meany += p.y;
	}

	meanx /= ptsNum;
	meany /= ptsNum;

	float spreadx = 0.0, spready = 0.0;
	for (int i = 0; i < ptsNum; i++)
	{
		auto& p = points[i];
		spreadx += (p.x - meanx) * (p.x - meanx);
		spready += (p.y - meany) * (p.y - meany);
	}

	spreadx /= ptsNum;
	spready /= ptsNum;

    spreadx += 1e-20f;
    spready += 1e-20f;

	cv::Mat offs = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat scale = cv::Mat::eye(3, 3, CV_32F);

	offs.at<float>(0, 2) = -meanx;
	offs.at<float>(1, 2) = -meany;

	const float sqrt2 = sqrt(2);

	scale.at<float>(0, 0) = sqrt2 / sqrt(spreadx);
	scale.at<float>(1, 1) = sqrt2 / sqrt(spready);

	NormalizedData result;
	result.T = scale * offs;

	for (int i = 0; i < ptsNum; i++)
	{
		cv::Point2f p1;

		auto& p = points[i];
		p1.x = sqrt2 * (p.x - meanx) / sqrt(spreadx);
		p1.y = sqrt2 * (p.y - meany) / sqrt(spready);

		result.points.emplace_back(p1);
	}

	return result;
}

cv::Mat calcHomography(std::vector<cv::Point2f>& points_map, std::vector<cv::Point2f>& points_image)
{

    const size_t ptsNum = points_image.size();
    cv::Mat A(2 * ptsNum, 6, CV_32F);


    for (int i = 0; i < ptsNum; i++)
    {
        float u1 = points_map[i].x;
        float v1 = points_map[i].y;

        float u2 = points_image[i].x;
        float v2 = points_image[i].y;

        A.at<float>(2 * i, 0) = v1;
        A.at<float>(2 * i, 1) = 1.0f;
        A.at<float>(2 * i, 2) = 0.0f;
        A.at<float>(2 * i, 3) = 0.0f;
        A.at<float>(2 * i, 4) = -u2 * v1;
        A.at<float>(2 * i, 5) = -u2;

        A.at<float>(2 * i + 1, 0) = 0.0f;
        A.at<float>(2 * i + 1, 1) = 0.0f;
        A.at<float>(2 * i + 1, 2) = v1;
        A.at<float>(2 * i + 1, 3) = 1;
        A.at<float>(2 * i + 1, 4) = -v2 * v1;
        A.at<float>(2 * i + 1, 5) = v2;
    }


    cv::Mat eVecs(6, 6, CV_32F), eVals(6, 6, CV_32F);
    cv::eigen(A.t() * A, eVals, eVecs);

    cv::Mat H(3, 2, CV_32F);

    std::cout << "evecs\n" << eVecs << "\n";
    /* std::cout << "evals\n" << eVals << "\n"; */

    int count = 0;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 2; j++){
            H.at<float>(i, j) = eVecs.at<float>(5, count);
            count++;
        }
    }


    // normalize
    H = H * (1.0 / H.at<float>(2, 1));

    std::cout << ":::::H:::::\n" << H  << "\n";
    
    return H;
    /* return normalized_points_map.T.inv() * H * normalized_points_image.T; */
}


int main(int argc, char* argv[]){

    std::ifstream infile("center_circle_pts.txt");
	if (!infile) {
		std::cout << "Unable to read file." << std::endl;
		return -1;
	}
    
    std::vector<cv::Point2f> points_image;

	std::string line;
	while (getline(infile, line)){
		std::istringstream ss(line);
		float x, y;
		ss >> x >> y;
        points_image.emplace_back(cv::Point2f(x, y));
	}

    NormalizedData points_image_nd = normalizeData(points_image);
    std::vector<cv::Point2f> points_image_normalized = points_image_nd.points;

    std::vector<cv::Point2f> points_image_line(points_image_normalized.begin(), points_image_normalized.begin()+3);
    std::vector<cv::Point2f> points_image_circle(points_image_normalized.begin()+3, points_image_normalized.end());

    /* for(auto & e: points_image_normalized) std::cout << e << "\n"; */

    std::vector<cv::Point2f> points_map;
    points_map.emplace_back(cv::Point2f(0, 1));
    points_map.emplace_back(cv::Point2f(0, 0));
    points_map.emplace_back(cv::Point2f(0, -1));

    NormalizedData points_map_nd = normalizeData(points_map);
    std::vector<cv::Point2f> points_map_normalized = points_map_nd.points;

    for(auto & e: points_image_line) std::cout << e << "\n";

    cv::Mat image = calcHomography(points_map_normalized, points_image_line);

    return 0;
}
