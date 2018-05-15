/*
 * Implements binarization of documents
 * Check this paper: DOI 10.1007/s10032-010-0142-4
 * Christian Wolf, Jean-Michel Jolion and Francoise Chassaing.
 * Text Localization, Enhancement and Binarization in Multimedia Documents.
 * International Conference on Pattern Recognition (ICPR),
 * volume 4, pages 1037-1040, 2002.
 */

#ifndef IMAGEBINARIZATION_H
#define IMAGEBINARIZATION_H

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include <time.h>

enum NiblackVersion {
    NIBLACK=0,
    SAUVOLA,
    WOLFJOLION,
};

class ImageBinarization
{
public:
    ImageBinarization();
    virtual ~ImageBinarization();
    void convertToBin(cv::Mat &bin, cv::Mat &image);
    void calcLocalStats(double &res, cv::Mat &im, cv::Mat &map_m, cv::Mat &map_s, int &winx, int &winy);
    void applyNiblackSauvolaWolfJolion(cv::Mat &im, cv::Mat &output, NiblackVersion version =WOLFJOLION, double k=0.5, double dR = 128);
};

#endif // IMAGEBINARIZATION_H
