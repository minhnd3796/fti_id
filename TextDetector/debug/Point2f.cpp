#include <iostream>
#include <opencv2/core/core.hpp>

int main()
{
    cv::Mat img = cv::Mat::zeros(10, 10, CV_8UC1);
    img.at<uchar>(3, 7) = 255;
    cv::Point2f p;
    p.x = 3;
    p.y = 7;
    std::cout << p << std::endl;
    return 0;
}
