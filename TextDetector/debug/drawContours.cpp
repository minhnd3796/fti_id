#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    cv::Mat img = cv::imread("round.png", cv::IMREAD_GRAYSCALE);
    cv::Mat thresh;
    cv::threshold(img, thresh, 127, 255, 0);
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    // find contour
    cv::findContours(thresh, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
    // cv::drawContours(thresh, contours, 0, 127, 3);
    std::vector<cv::RotatedRect> min_rects;
    for (int i = 0; i < (int)contours.size(); i++) {
        // cv::drawContours(thresh, contours, i, 127, 3);
        min_rects.push_back(cv::minAreaRect(cv::Mat(contours[i])));
    }
    for(cv::RotatedRect rect : min_rects)
    {
        // cv::drawContours(dilated_bin_img, contours, i, 127, 1);
        cv::Point2f rect_points[4]; rect.points(rect_points);
        for(int j = 0; j < 4; j++)
        {
            std::cout << "(" << rect_points[j].y << ", " << rect_points[j].x << ")" <<  std::endl;
            cv::line(thresh, rect_points[j], rect_points[(j + 1) % 4], 127, 1, 8);
        }
    }

    // Display visualised image
    cv::namedWindow("Debug Window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Debug Window", thresh);
    cv::waitKey(0);
}