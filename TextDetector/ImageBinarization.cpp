#include "ImageBinarization.h"

ImageBinarization::ImageBinarization()
{

}

ImageBinarization::~ImageBinarization() {

}

void ImageBinarization::convertToBin(cv::Mat &bin_image, cv::Mat &image) {
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    int ddepth = CV_16S;
    int ksize = 3;
    int scale = 1;
    int delta = 0;

    cv::Sobel(image, grad_x, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(image, grad_y, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);

    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, bin_image);

    // binarize the image
    cv::threshold(bin_image, bin_image, 32, 255, CV_THRESH_BINARY);
}

void ImageBinarization::calcLocalStats(double &res, cv::Mat &im, cv::Mat &map_m, cv::Mat &map_s, int &winx, int &winy) {
    double m,s,max_s, sum, sum_sq, foo;
    int wxh	= winx/2;
    int wyh	= winy/2;
    int x_firstth= wxh;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    double winarea = winx*winy;

    max_s = 0;
    for	(int j = y_firstth ; j<=y_lastth; j++)
    {
        // Calculate the initial window at the beginning of the line
        sum = sum_sq = 0;
        for	(int wy=0 ; wy<winy; wy++)
            for	(int wx=0 ; wx<winx; wx++) {
                foo = im.at<uchar>(j-wyh+wy,wx);
                sum    += foo;
                sum_sq += foo*foo;
            }
        m  = sum / winarea;
        s  = sqrt ((sum_sq - (sum*sum)/winarea)/winarea);
        if (s > max_s)
            max_s = s;
        map_m.at<uchar>( j,x_firstth) = m;
        map_s.at<uchar>( j,x_firstth) = s;

        // Shift the window, add and remove	new/old values to the histogram
        for	(int i=1 ; i <= im.cols-winx; i++) {

            // Remove the left old column and add the right new column
            for (int wy=0; wy<winy; ++wy) {
                foo = im.at<uchar>(j-wyh+wy,i-1);
                sum    -= foo;
                sum_sq -= foo*foo;
                foo = im.at<uchar>(j-wyh+wy,i+winx-1);
                sum    += foo;
                sum_sq += foo*foo;
            }
            m  = sum / winarea;
            s  = sqrt ((sum_sq - (sum*sum)/winarea)/winarea);
            if (s > max_s)
                max_s = s;
            map_m.at<uchar>( j,i+wxh) = m;
            map_s.at<uchar>( j,i+wxh) = s;
        }
    }

    res = max_s;
}

void ImageBinarization::applyNiblackSauvolaWolfJolion(cv::Mat &im, cv::Mat &output, NiblackVersion version, double k, double dR) {
    int winy = (int) (2.0 * im.rows-1)/3;
    int winx = (int) im.cols-1 < winy ? im.cols-1 : winy;
    if (winx > 100) winx = winy = 40;

    double m, s, max_s;
    double th=0;
    double min_I, max_I;
    int wxh	= winx/2;
    int wyh	= winy/2;
    int x_firstth= wxh;
    int x_lastth = im.cols-wxh-1;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;

    // Create local statistics and store them in a double matrices
    cv::Mat map_m = cv::Mat::zeros (im.rows, im.cols, CV_32F);
    cv::Mat map_s = cv::Mat::zeros (im.rows, im.cols, CV_32F);
    max_s = 0;
    this->calcLocalStats (max_s, im, map_m, map_s, winx, winy);

    cv::minMaxLoc(im, &min_I, &max_I);

    cv::Mat thsurf (im.rows, im.cols, CV_32F);

    // Create the threshold surface, including border processing
    // ----------------------------------------------------

    for	(int j = y_firstth ; j<=y_lastth; j++) {

        // NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
        for	(int i=0 ; i <= im.cols-winx; i++) {

            m  = map_m.at<uchar>( j,i+wxh);
            s  = map_s.at<uchar>( j,i+wxh);

            // Calculate the threshold
            switch (version) {

            case NIBLACK:
                th = m + k*s;
                break;

            case SAUVOLA:
                th = m * (1 + k*(s/dR-1));
                break;

            case WOLFJOLION:
                th = m + k * (s/max_s-1) * (m-min_I);
                break;

            default:
                //cerr << "Unknown threshold type in ImageThresholder::surfaceNiblackImproved()\n";
                exit (1);
            }

            thsurf.at<uchar>(j,i+wxh) = th;

            if (i==0) {
                // LEFT BORDER
                for (int i=0; i<=x_firstth; ++i)
                    thsurf.at<uchar>(j,i) = th;

                // LEFT-UPPER CORNER
                if (j==y_firstth)
                    for (int u=0; u<y_firstth; ++u)
                        for (int i=0; i<=x_firstth; ++i)
                            thsurf.at<uchar>(u,i) = th;

                // LEFT-LOWER CORNER
                if (j==y_lastth)
                    for (int u=y_lastth+1; u<im.rows; ++u)
                        for (int i=0; i<=x_firstth; ++i)
                            thsurf.at<uchar>(u,i) = th;
            }

            // UPPER BORDER
            if (j==y_firstth)
                for (int u=0; u<y_firstth; ++u)
                    thsurf.at<uchar>(u,i+wxh) = th;

            // LOWER BORDER
            if (j==y_lastth)
                for (int u=y_lastth+1; u<im.rows; ++u)
                    thsurf.at<uchar>(u,i+wxh) = th;
        }

        // RIGHT BORDER
        for (int i=x_lastth; i<im.cols; ++i)
            thsurf.at<uchar>(j,i) = th;

        // RIGHT-UPPER CORNER
        if (j==y_firstth)
            for (int u=0; u<y_firstth; ++u)
                for (int i=x_lastth; i<im.cols; ++i)
                    thsurf.at<uchar>(u,i) = th;

        // RIGHT-LOWER CORNER
        if (j==y_lastth)
            for (int u=y_lastth+1; u<im.rows; ++u)
                for (int i=x_lastth; i<im.cols; ++i)
                    thsurf.at<uchar>(u,i) = th;
    }
    //cerr << "surface created" << endl;


    for	(int y=0; y<im.rows; ++y)
        for	(int x=0; x<im.cols; ++x)
        {
            if (im.at<uchar>(y,x) >= thsurf.at<uchar>(y,x))
            {
                output.at<uchar>(y,x) = 255;
            }
            else
            {
                output.at<uchar>(y,x) = 0;
            }
        }
}
