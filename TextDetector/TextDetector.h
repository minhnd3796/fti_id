#ifndef TEXTDETECTOR_H
#define TEXTDETECTOR_H

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "ImageBinarization.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <istream>

#include <chrono>
#include <ctime>
#include "Singleton.h"

#define _TINY_UTF8_H_USE_IOSTREAM_
#include "tinyutf8.h"

#define MAX_H_RATIO 8.82
#define MIN_H_RATIO 79.55

#define MIN_H_POSPROCESSING_RATIO 30

#define MAX_SIZE_RATIO 800.586
#define MIN_SIZE_RATIO 90500.14

#define X_IMG_RATIO 0.306
#define Y_IMG_RATIO 0.891

#define Y_BOT_RATIO 0.9803
#define X_RIGHT_RATIO 0.9803

#define Y_S1_REGION 2.976562

#define POINT_CENTER_DELTA 27.32

#define MIN_H_ROTATED_RECT_POST_RATIO 25.93

#define DELTA_H_RATIO   6

#define CARD_ID_Y_MIN_RATIO 4.1
#define CARD_ID_X_MIN_RATIO 3.107
#define CARD_ID_Y_MAX_RATIO 2.7848
#define CARD_ID_X_MAX_RATIO 1.025

#define CARD_ID_MIN_H   2.8

#define CARD_NAME_Y_MIN_RATIO 2.84328
#define CARD_NAME_X_MIN_RATIO 2.38568
#define CARD_NAME_Y_MAX_RATIO 1.95385
#define CARD_NAME_X_MAX_RATIO 1.04167

#define CARD_DOB_Y_MIN_RATIO 1.8343
#define CARD_DOB_Y_MAX_RATIO 1.7
#define CARD_DOB_X_MIN_RATIO 2.111
#define CARD_DOB_X_MAX_RATIO 1.0832

#define CARD_ADDR_Y_MIN_RATIO 3.42667
#define CARD_ADDR_Y_MAX_RATIO 1.028
#define CARD_ADDR_X_MIN_RATIO 1.5811
#define CARD_ADDR_X_MAX_RATIO 1.06874

#define NQ_Y_MIN_RATIO 1.5257
#define NQ_Y_MAX_RATIO 1.4841
#define NQ_X_MIN_RATIO 3.13414
#define NQ_X_MAX_RATIO 2.03162

#define NDKTT_Y_MIN_RATIO 1.21375
#define NDKTT_Y_MAX_RATIO 1.18727
#define NDKTT_X_MIN_RATIO 3.13414
#define NDKTT_X_MAX_RATIO 1.66883

enum COLOR_FILTER_TYPE {
    RED = 1
};

enum ID_TYPE {
    FRONT = 0,
    BACK
};

class TextDetector: public Singleton<TextDetector>
{
    friend class Singleton<TextDetector>;
public:
    TextDetector();
    virtual ~TextDetector();

    void bgrToHSV(double b, double g, double r, double &h, double &s, double &v);
    void bgrToHSV(cv::Mat &hsv, cv::Mat &bgr);
    void colorFilter(cv::Mat &res, cv::Mat &input, int color_code);
    void convertToGrayscale(cv::Mat &gray_img, cv::Mat &input, double ratio = 0.5);
    void applyBinarizationFilter(cv::Mat &bin_img, cv::Mat &gray, int method, double k = 0.5, double dR = 128);
    void removeUnwantedComponents(cv::Mat &res, cv::Mat &input, int &min, int &max);
    void dilateImage(cv::Mat &res, cv::Mat &img, int &h_size);
    void getMinRectangles(std::vector<cv::RotatedRect> &min_rects, std::vector<std::vector<cv::Point> > &contours, cv::Mat &bin_img);

    void minRectFilter1(cv::Mat &img, cv::Mat &mask, std::vector<cv::RotatedRect> &min_rects);
    void minRectFilter2(cv::Mat &img, std::vector<cv::RotatedRect> &min_rects);
    std::vector<std::string> split(const std::string &s, char delim);
    static bool sortFunc(const std::pair<float, float> &A, const std::pair<float, float> &B);
    void computeDistance(float &distance, cv::Point2f &p1, cv::Point2f &p2);
    void groupByLine(std::vector<std::vector<cv::RotatedRect> > &mlines, std::vector<cv::RotatedRect> &min_rects, int &h);
    void groupByLine(std::vector<int> &y_index, std::vector<std::vector<cv::RotatedRect> > &lines, std::vector<cv::RotatedRect> &min_rects, int &h);
    void sortByX(std::vector<int> &indexs, std::vector<cv::RotatedRect> &min_rects);
    void sortByXY(std::vector<int> &indexs, std::vector<cv::RotatedRect> &min_rects);
    void saveParts(std::vector<cv::RotatedRect> &min_rects, cv::Mat &img, std::string &output, std::string &file_name);
    void saveParts(std::vector<int> &indexs, std::vector<cv::RotatedRect> &min_rects, cv::Mat &img, std::string &output, std::string &file_name);
    void saveParts(std::vector<std::vector<cv::RotatedRect> > &lines, cv::Mat &img, std::string &output, std::string &file_name);
    void saveNameField(std::vector<cv::RotatedRect> &name_field, std::vector<std::vector<cv::RotatedRect> > &lines, cv::Mat &img, std::string &output, std::string &file_name);
    void saveDoBField(std::vector<cv::RotatedRect> &dob_field, std::vector<std::vector<cv::RotatedRect> > &lines, cv::Mat &img);
    void saveAddrField(std::vector<cv::RotatedRect> &addr_field, std::vector<std::vector<cv::RotatedRect> > &lines, cv::Mat &img);
    void saveIDAreaToImg(cv::Mat &res, cv::Mat &img_bin, cv::Mat &img);
    void saveDoBAreaToImg(cv::Mat &res, std::vector<cv::RotatedRect> &min_rects, cv::Mat &img);

    void visualizeRotatedRectsAndContours(cv::Mat &res, std::vector<cv::RotatedRect> &min_rects, std::vector<std::vector<cv::Point> > &contours);
    void visualizeRotatedRectsByLine(cv::Mat &res, std::vector<std::vector<cv::RotatedRect> > &lines);

    void processImage(cv::Mat &img, cv::Mat &visualized_img, std::vector<cv::Rect> &rects, int type, bool is_used_filter);
    void saveBoxesToFile(std::string &name_prefix, std::vector<cv::Rect> &rects, cv::Mat &img);
};

#endif // TEXTDETECTOR_H
