#define MAX_SIZE_RATIO 800.586
#define MIN_SIZE_RATIO 90500.14

#define X_IMG_RATIO 0.306
#define Y_IMG_RATIO 0.891

#define Y_BOT_RATIO 0.9803
#define X_RIGHT_RATIO 0.9803

#define POINT_CENTER_DELTA 27.32

#define MAX_H_RATIO 8.82
#define MIN_H_RATIO 79.55

#define MIN_H_POSPROCESSING_RATIO 30

#define Y_S1_REGION 2.976562

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "../BlobsLib/BlobResult.h"

enum ID_TYPE {
    FRONT = 0,
    BACK
};

enum NiblackVersion {
    NIBLACK=0,
    SAUVOLA,
    WOLFJOLION,
};

cv::Mat removeUnwantedComponents(cv::Mat &input, int &min, int &max) {
    CBlobResult blobs = CBlobResult(input, cv::Mat(), 4);
    blobs.Filter(blobs, B_INCLUDE, CBlobGetLength(), B_GREATER, min);
    blobs.Filter(blobs, B_EXCLUDE, CBlobGetLength(), B_GREATER, max);
    cv::Mat res = cv::Mat::zeros(input.size(), input.type());
    for (int i = 0; i < blobs.GetNumBlobs(); i++) {
        blobs.GetBlob(i)->FillBlob(res, CV_RGB(255,255,255));
    }
    return res;
}

void dilateImage(cv::Mat &res, cv::Mat &img, int &h_size) {
    cv::Mat h_structure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(h_size,1));
    cv::dilate(img, res, h_structure);
}

void getMinRectangles(std::vector<cv::RotatedRect> &min_rects, std::vector<std::vector<cv::Point> > &contours, cv::Mat &bin_img) {
    std::vector<cv::Vec4i> hierarchy;
    
    // find contour
    cv::findContours(bin_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
    for (int i = 0; i < (int)contours.size(); i++) {
        min_rects.push_back(cv::minAreaRect(cv::Mat(contours[i])));
    }
}

void sortByX(std::vector<int> &indexs, std::vector<cv::RotatedRect> &min_rects) {
    std::vector<float> vals;
    indexs.clear();
    for (int i = 0; i < (int) min_rects.size(); i++) {
        vals.push_back(min_rects[i].center.x);
        indexs.push_back(i);
    }

    float swap_val = 0;
    float swap_idx = 0;
    for (int i = 0; i < (int)min_rects.size(); i++) {
        for (int j = i; j < (int)min_rects.size(); j++) {
            if (vals[i] > vals[j]) {
                swap_val = vals[i];
                vals[i] = vals[j];
                vals[j] = swap_val;

                swap_idx = indexs[i];
                indexs[i] = indexs[j];
                indexs[j] = swap_idx;
            }
        }
    }
}

void groupByLine(std::vector<std::vector<cv::RotatedRect> > &mlines, std::vector<cv::RotatedRect> &min_rects, int &h) {
    std::vector<std::vector<cv::RotatedRect> > lines;

    float delta_h = (float)h / POINT_CENTER_DELTA;
    for (int i = 0; i < (int)min_rects.size(); i++) {
        cv::RotatedRect rect = min_rects[i];
        std::vector<cv::RotatedRect> line;
        if (rect.center == cv::Point2f(-1, -1)) continue;
        for (int j = 0; j < (int)min_rects.size(); j++) {
            if (min_rects[j].center == cv::Point2f(-1, -1)) continue;
            if (std::abs(rect.center.y - min_rects[j].center.y) < delta_h) {
                line.push_back(min_rects[j]);
                min_rects[j].center = cv::Point2f(-1, -1);
            }
        }
        if (line.size() > 0) {
            std::vector<cv::RotatedRect> mline;
            std::vector<int> mindex;
            sortByX(mindex, line);
            for (int j = 0; j < (int)mindex.size(); j++) {
                mline.push_back(line[mindex[j]]);
            }
            lines.push_back(mline);
        }
    }

    std::vector<int> y_index;
    std::vector<float> vals;
    y_index.clear();
    for (int i = 0; i < (int)lines.size(); i++) {
        vals.push_back(lines[i][0].center.y);
        y_index.push_back(i);
    }

    int swap_idx = 0;
    float swap_val = 0;
    for (int i = 0; i < (int)vals.size(); i++) {
        for (int j = i; j < (int)vals.size(); j++) {
            if (vals[i] > vals[j]) {
                swap_val = vals[i];
                vals[i] = vals[j];
                vals[j] = swap_val;

                swap_idx = y_index[i];
                y_index[i] = y_index[j];
                y_index[j] = swap_idx;
            }
        }
    }

    for (int i = 0; i < (int)lines.size(); i++) {
        std::vector<cv::RotatedRect> line;
        for (int j = 0; j < (int)lines[y_index[i]].size(); j++) {
            line.push_back(lines[y_index[i]][j]);
        }
        mlines.push_back(line);
    }
}

void visualizeRotatedRectsByLine(cv::Mat &res, std::vector<std::vector<cv::RotatedRect> > &lines) {
    cv::Scalar color;
    cv::Scalar color2 = cv::Scalar(0,255,0);
    cv::Point2f rect_points[4];
    for (int i = 0; i <(int) lines.size(); i++) {
        if (i%2) color = cv::Scalar(255,0,255);
        else color = cv::Scalar(255,0,0);
        for (int j = 0; j <(int)lines[i].size(); j++) {
            lines[i][j].points(rect_points);
            // std::string pos_text = "[" + std::to_string((int)lines[i][j].center.x) + ","+ std::to_string((int)lines[i][j].center.y) + "]";
            cv::circle(res, lines[i][j].center,3, color2, 2);
            for (int j = 0; j < 4; j++) {
                cv::line(res, rect_points[j], rect_points[(j+1)%4], color, 2, 8);
            }
            // cv::putText(res, pos_text, lines[i][j].center, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0,255,255),2);
        }
    }
    // visualize regions
}

void minRectFilter1(cv::Mat &img, cv::Mat &mask, std::vector<cv::RotatedRect> &min_rects) {
    float min_h = (float) img.rows / MIN_H_RATIO;
    float max_h = (float) img.rows / MAX_H_RATIO;
    float max_w = (float) img.cols * X_RIGHT_RATIO;

    cv::Point2f p1, p2;
    // p1 is filter of image region
    p1.x = (float)img.cols * X_IMG_RATIO;
    p1.y = (float)img.rows * Y_IMG_RATIO;

    // p2 is botom region
    p2.y = (float)img.rows * Y_BOT_RATIO;
    for (int i = 0; i < (int)min_rects.size(); i++) {
        cv::Rect brect = min_rects[i].boundingRect();
        if (brect.height < min_h) {
            min_rects[i].center = cv::Point2f(-1,-1);
            continue;
        }
        if (brect.height > max_h) {
            min_rects[i].center = cv::Point2f(-1,-1);
            continue;
        }

        if (min_rects[i].center.x < p1.x && min_rects[i].center.y < p1.y) {
            min_rects[i].center = cv::Point2f(-1,-1);
            continue;
        }

        if (min_rects[i].center.y > p2.y) {
            min_rects[i].center = cv::Point2f(-1,-1);
            continue;
        }
        if (min_rects[i].center.x > max_w) {
            min_rects[i].center = cv::Point2f(-1,-1);
            continue;
        }

        if (brect.x < 0) {
            brect.width += brect.x;
            brect.x = 0;
        }
        if (brect.y < 0) {
            brect.height += brect.y;
            brect.y = 0;
        }
        if (brect.x + brect.width >= img.cols) {
            brect.width = img.cols - brect.x - 1;
        }
        if (brect.y + brect.height >= img.rows) {
            brect.height = img.rows - brect.y - 1;
        }

        cv::Point2f vertices[4];
        min_rects[i].points(vertices);
        for (int j = 0; j < 4; j++) {
            vertices[j].x -= brect.x;
            vertices[j].y -= brect.y;
        }
        cv::Mat m_mask = cv::Mat(mask, brect);
        cv::Mat local_mask = cv::Mat::zeros(m_mask.size(), m_mask.type());
        for (int j = 0; j < 4; j++) {
            cv::line(local_mask, vertices[j], vertices[(j+1)%4], cv::Scalar(255,255,255));
        }

        int first = -1, last = -1;
        int count = 0;

        std::vector<cv::Point> points;
        for (int y = 0; y < local_mask.rows; y++) {
            for (int x = 0; x < local_mask.cols; x++) {
                if (local_mask.at<cv::Vec3b>(y,x) == cv::Vec3b(255,255,255)) {
                    if (first != -1) {
                        last = x;
                        count = 2;
                    } else {
                        first = x;
                        count = 1;
                    }

                }
                if (x == local_mask.cols -1) {
                    cv::Point p;
                    if (count == 1) {
                        p.x = first;
                        p.y = y;
                        points.push_back(p);
                    }
                    if (count == 2) {
                        for (int i = first; i <= last; i++) {
                            p.x = i;
                            p.y = y;
                            points.push_back(p);
                        }
                    }
                    first = -1;
                    last = -1;
                    count = 0;
                }

            }
        }

        for (int n = 0; n < (int)points.size(); n ++) {
            m_mask.at<cv::Vec3b>(points[n]) = cv::Vec3b(255,255,255);
        }
        cv::imwrite("m_mask/m_mask_" + std::to_string(i) +".png", m_mask);
        cv::imwrite("local_mask/local_mask_" + std::to_string(i) +".png", local_mask);
        cv::imwrite("full_mask/full_mask_" + std::to_string(i) +".png", mask);
        // std::cout << std::to_string(i) + ": [" << brect.width << " x " << brect.height << "]" << m_mask.size() << local_mask.size() << std::endl;
    }
}

void minRectFilter2(cv::Mat &img, std::vector<cv::RotatedRect> &min_rects) {
    int min_h_pos = (float)img.rows/ MIN_H_POSPROCESSING_RATIO;
    float min_y1 = (float) img.rows / Y_S1_REGION;
    for (int i = 0; i < (int) min_rects.size(); i++) {
        cv::Rect brect = min_rects[i].boundingRect();
        if (brect.height <= min_h_pos) {
            min_rects[i].center = cv::Point2f(-1, -1);
            continue;
        }
        if (min_rects[i].center.y < min_y1) {
            min_rects[i].center = cv::Point2f(-1, -1);
            continue;
        }
    }
}

void convertToGrayscale(cv::Mat &gray_img, cv::Mat &input, double ratio) {
    gray_img = cv::Mat::zeros(input.size(), CV_8UC1);
    for (int y = 0; y < gray_img.rows; y++) {
        for (int x = 0; x < gray_img.cols; x++) {
            cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
            int max = pixel[0];
            int min = pixel[0];
            if (pixel[1] > pixel[0]) max = pixel[1];
            else max = pixel[0];
            if (max < pixel[2]) max = pixel[2];
            else {
                if (min > pixel[2]) min = pixel[2];
            }
            gray_img.at<uchar>(y, x) = (uchar)((double)max * ratio + (double)min *(1.0 - ratio));
        }
    }
}

void calcLocalStats(double &res, cv::Mat &im, cv::Mat &map_m, cv::Mat &map_s, int &winx, int &winy) {
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
        m = sum / winarea;
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

void applyNiblackSauvolaWolfJolion(cv::Mat &im, cv::Mat &output, NiblackVersion version, double k, double dR) {
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
    calcLocalStats(max_s, im, map_m, map_s, winx, winy);

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

cv::Mat applyBinarizationFilter(cv::Mat &gray, int method, double k, double dR) {
    NiblackVersion version = (NiblackVersion) method;
    cv::Mat bin_img = cv::Mat::zeros(gray.size(), gray.type());
    applyNiblackSauvolaWolfJolion(gray, bin_img, version, k, dR);
    return bin_img;
}

void processImage(cv::Mat &img, cv::Mat &visualised_img, std::vector<cv::Rect> &rects, int type, bool is_used_filter) {
    rects.clear();
    cv::Mat gray;
    convertToGrayscale(gray, img, 0.5);
    cv::Mat bin_img = applyBinarizationFilter(gray, 1, 0.35, 100);   
    
    // compute min_h, max_h, based on input size
    float img_size = (float)img.rows * (float)img.cols;
    int min_size = img_size / MIN_SIZE_RATIO, max_size = img_size / MAX_SIZE_RATIO;

    cv::Mat inv_bin_img = cv::Scalar::all(255) - bin_img;
    cv::Mat filtered_bin_img = removeUnwantedComponents(inv_bin_img, min_size, max_size);
    for (int y = 0; y < inv_bin_img.rows; y++) {
        for (int x = 0; x < inv_bin_img.cols; x++) {
            if (inv_bin_img.at<uchar>(y,x) == 0) {
                filtered_bin_img.at<uchar>(y,x) = 0;
            }
        }
    }
    cv::Mat dilated_bin_img;
    int h_size = 5;
    dilateImage(dilated_bin_img, filtered_bin_img, h_size);


    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::RotatedRect> min_rects;

    getMinRectangles(min_rects, contours, dilated_bin_img);
    
    visualised_img = img.clone();

    if (!is_used_filter) {
        std::vector<std::vector<cv::RotatedRect> > lines;
        groupByLine(lines, min_rects, img.rows);
        visualizeRotatedRectsByLine(visualised_img, lines);
        for (size_t i = 0; i < lines.size(); i++) {
            for (size_t j = 0; j < lines[i].size(); j++) {
                cv::Rect rect = lines[i][j].boundingRect();
                rects.push_back(rect);
            }
        }
        return;
    }

    // create a mask from min_rect and then find another min_rect
    cv::Mat full_mask = cv::Mat::zeros(img.size(), CV_8UC3);
    if (type == FRONT) {
        // for ID holder info area
        minRectFilter1(img, full_mask, min_rects);

        cv::Mat full_mask_bin;
        cv::cvtColor(full_mask, full_mask_bin, CV_BGR2GRAY);
        std::vector<std::vector<cv::Point> > contours2;
        std::vector<cv::RotatedRect> min_rects2;
        getMinRectangles(min_rects2, contours2, full_mask_bin);
        // Display visualised image
        // cv::namedWindow("Debug Window", cv::WINDOW_AUTOSIZE);
        // cv::imshow("Debug Window", full_mask_bin);
        minRectFilter2(img, min_rects2);

        if (is_used_filter) {
            std::vector<std::vector<cv::RotatedRect> > lines;
            groupByLine(lines, min_rects2, img.rows);
            visualizeRotatedRectsByLine(visualised_img, lines);
            for (size_t i = 0; i < lines.size(); i++) {
                for (size_t j = 0; j < lines[i].size(); j++) {
                    cv::Rect rect = lines[i][j].boundingRect();
                    rects.push_back(rect);
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    cv::Mat input_img = cv::imread("test.jpg");
    std::string prefix = std::string("test");
    cv::Mat visualised_img;
    std::vector<cv::Rect> rects;
    processImage(input_img, visualised_img, rects, FRONT, true);
    std::string visualised_file = prefix +"_visualised.png";
    cv::imwrite(visualised_file, visualised_img);

    // Display visualised image
    // cv::namedWindow("Debug Window", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Debug Window", visualised_img);
    cv::waitKey(0);
}
