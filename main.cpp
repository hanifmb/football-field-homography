#include <vector>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include <fstream>
#include <sstream>

#include <unsupported/Eigen/NonLinearOptimization>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cmath>

typedef std::vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d> > Point2DVector;

Point2DVector GeneratePoints();

// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
typedef _Scalar Scalar;
enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
};
typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

int m_inputs, m_values;

Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

int inputs() const { return m_inputs; }
int values() const { return m_values; }

};


struct MyFunctor : Functor<double>
{
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
  {
    // "a" in the model is x(0), and "b" is x(1)
    for(unsigned int i = 0; i < this->Points.size(); ++i)
      {
        float u2 = this->Points[i](0);
        float v2 = this->Points[i](1);

        float alpha = x(0);
        float beta = x(1);
        float gamma = x(2);
        
        float h2 = 0.59885508;
        float h3 = -0.037879128;
        float h5 = 0.83761203;
        float h6 = 0.25667429;
        float h8 = -0.031710938;

        float D1 = h5 - h8 * h6;
        float D4 = h8 * h3 - h2;
        float D7 = h2 * h6 - h5 * h3;
        
        float a5 = 1;
        float a6 = -h8;
        float a8 = -h6;
        float a9 = h5;
        float b2 = -1;
        float b3 = h8;
        float b8 = h3;
        float b9 = -h2;
        float c2 = h6;
        float c3 = -h5;
        float c5 = -h3;
        float c6 = h2;

        float A2 = a5 * v2 + a8;
        float A3 = a6 * v2 + a9;
        float B2 = b2 * u2 + b8;
        float B3 = b3 * u2 + b9;
        float C2 = c2 * u2 + c5 * v2;
        float C3 = c3 * u2 + c6 * v2;
        
        float D = D1 * u2 + D4 * v2 + D7;
        float R = 1.9945;

        fvec(i) = pow(D, 2) + pow(A2 * alpha + B2 * beta + C2 * gamma, 2) - pow(R, 2) * pow(A3 * alpha + B3 * beta + C3 * gamma, 2);
      /* fvec(i) = this->Points[i](1) - (x(0) * this->Points[i](0) + x(1)); */
      /* fvec(i) = this->Points[i](1) - (x(0) * this->Points[i](0) + x(1)); */
      }

    return 0;
  }

  Point2DVector Points;
  
  int inputs() const { return 3; } // There are two parameters of the model
  int values() const { return this->Points.size(); } // The number of observations
};

struct MyFunctorNumericalDiff : Eigen::NumericalDiff<MyFunctor> {};

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

    /* spreadx += 1e-20f; */
    /* spready += 1e-20f; */

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

    std::cout << "evecs:\n" << eVecs << "\n";
    std::cout << "evals:\n" << eVals << "\n";

    int count = 0;
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 2; j++){
            H.at<float>(i, j) = eVecs.at<float>(5, count);
            count++;
        }
    }

    // normalize
    H = H * (1.0 / H.at<float>(2, 1));
    
    return H;
    /* return normalized_points_map.T.inv() * H * normalized_points_image.T; */
}

void draw_cross(cv::Point pt, cv::Mat image, int size){

        cv::Point starting1(pt.x - size, pt.y-size);
        cv::Point ending1(pt.x + size, pt.y+size);

        cv::Point starting2(pt.x + size, pt.y-size);
        cv::Point ending2(pt.x-size, pt.y+size);

        cv::Scalar line_Color(255, 255, 255);
       int thickness = 2;
       
    cv::line(image, starting1, ending1, line_Color, thickness);
    cv::line(image, starting2, ending2, line_Color, thickness);
}

void transformPoint(const cv::Point2f& input, cv::Point2f& output, const cv::Mat& H, bool isPerspective)
{
    cv::Mat pt(3, 1, CV_32FC1);
    pt.at<float>(0, 0) = input.x;
    pt.at<float>(1, 0) = input.y;
    pt.at<float>(2, 0) = 1.0;
    
    cv::Mat ptTransformed = H * pt;
    if (isPerspective)
        ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed;

    float newX = (ptTransformed.at<float>(0, 0));
    float newY = (ptTransformed.at<float>(1, 0));

    output = cv::Point2f(newX, newY);
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

    std::vector<cv::Point2f> points_map;
    points_map.emplace_back(cv::Point2f(0, 1));
    points_map.emplace_back(cv::Point2f(0, 0));
    points_map.emplace_back(cv::Point2f(0, -1));

    for(int i = 1; i < 360; i++){
        float angle = static_cast<float>(i) * 3.14159 / 180.0;
        float x = cos(angle);
        float y = sin(angle);
        points_map.emplace_back(cv::Point2f(x, y));
    }

    NormalizedData points_map_nd = normalizeData(points_map);
    std::vector<cv::Point2f> points_map_normalized = points_map_nd.points;
    
    std::vector<cv::Point2f> points_map_line(points_map_normalized.begin(), points_map_normalized.begin()+3);

    std::cout << "points_image_line:\n"; 
    for(auto & e: points_image_line) std::cout << e << "\n";
    
    std::cout << "points_map_line:\n"; 
    for(auto & e: points_map_line) std::cout << e << "\n";

    cv::Mat H = calcHomography(points_map_line, points_image_line);

    std::cout << "6 H Normalized:\n" << H << "\n";

    //calculating the last three homography elements
    Eigen::VectorXd x(3);
    /* x.fill(2.0f); */
    x(0) = 0;
    x(1) = 0;
    x(2) = 0;

    Point2DVector points;

    for(auto & e: points_image_circle){
        Eigen::Vector2d point;
        point(0) = e.x;
        point(1) = e.y;
        points.push_back(point);
    }
        
    MyFunctorNumericalDiff functor;
    functor.Points = points;
    Eigen::LevenbergMarquardt<MyFunctorNumericalDiff> lm(functor);

    Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);
    std::cout << "status: " << status << std::endl;

    //std::cout << "info: " << lm.info() << std::endl;

    std::cout << "x that minimizes the function: " << std::endl << x << std::endl;
    
    cv::Mat final_H(3, 3, CV_32F);
    
    cv::Mat free_param(3, 1, CV_32F);
    free_param.at<float>(0, 0) = x(0);
    free_param.at<float>(1, 0) = x(1);
    free_param.at<float>(2, 0) = x(2);

    hconcat(free_param, H, final_H);
    
    std::cout << "9H Normalized\n" << final_H << "\n";

    final_H = points_image_nd.T.inv() * final_H * points_map_nd.T;
    std::cout << "9H Denormalized by T\n" << final_H << "\n";
    
    //Generate unit circle data 
    std::vector<cv::Point2f> unit_circle;
    for(int i = 1; i < 360; i++){
        float angle = static_cast<float>(i) * 3.14159 / 180.0;
        float x = cos(angle);
        float y = sin(angle);
        unit_circle.emplace_back(cv::Point2f(x, y));
    }

    //projection to the original image
    cv::Mat img = cv::imread("image.png", cv::IMREAD_COLOR);
    cv::namedWindow("image", cv::WINDOW_NORMAL);

    cv::Point2f output;
    std::ofstream myfile("output.txt");
    for(auto & e:  unit_circle){

        transformPoint(e, output, final_H, true);
        myfile<< output.x << " " << output.y << std::endl;
        draw_cross(output, img, 15);

    }

    myfile.close();
    cv::imshow("image", img);
    cv::waitKey(0);

    return 0;
}
