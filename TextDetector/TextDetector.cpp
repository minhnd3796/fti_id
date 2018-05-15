#include "TextDetector.h"
#include "BlobsLib/BlobResult.h"

TextDetector::TextDetector()
{

}

TextDetector::~TextDetector() {

}


void TextDetector::bgrToHSV(double b, double g, double r, double &h, double &s, double &v) {
    double mnv, mxv, delta;

    mnv = mxv = r;
    if (mnv > g) mnv = g;
    if (mnv > b) mnv = b;
    if (mxv < g) mxv = g;
    if (mxv < b) mxv = b;
    v = mxv;

    delta = mxv - mnv;

    if ( mxv != 0.0)
        s = delta / mxv;
    else {
        // r = g = b = 0		// s = 0, v is undefined
        s = 0.0;
        h = -1.0;
        return;
    }

    if (std::fabs(r - mxv) < 1e-6)
        h = (g - b ) / delta;		// between yellow & magenta
    else if (std::fabs(g - mxv) < 1e-6)
        h = 2 + ( b - r ) / delta;	// between cyan & yellow
    else
        h = 4 + ( r - g ) / delta;	// between magenta & cyan

    h *= 60;				// degrees
    if (h < 0.0)
        h += 360;
    h /= 360.0;
}

void TextDetector::bgrToHSV(cv::Mat &hsv, cv::Mat &bgr) {
    hsv = cv::Mat::zeros(bgr.rows, bgr.cols, CV_32FC3);
    for (int i = 0; i < bgr.rows; ++i)
        for (int j = 0; j < bgr.cols; ++j) {
            double b = (double)bgr.at<cv::Vec3b>(i, j)[0];
            double g = (double)bgr.at<cv::Vec3b>(i, j)[1];
            double r = (double)bgr.at<cv::Vec3b>(i, j)[2];
            double h, s, v;
            this->bgrToHSV(b, g, r, h, s, v);
            hsv.at<cv::Vec3f>(i, j)[0] = h;
            hsv.at<cv::Vec3f>(i, j)[1] = s;
            hsv.at<cv::Vec3f>(i, j)[2] = v;
        }
}

void TextDetector::colorFilter(cv::Mat &res, cv::Mat &input, int color_code) {
    res = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
    cv::Mat hsv;
    this->bgrToHSV(hsv, input);
    for(int r = 0; r < hsv.rows; ++r)
        for(int c = 0; c < hsv.cols; ++c)
            //if ((c > hsv.cols/3) && (r < hsv.rows/2.3))
            if(color_code == RED) {
                if (hsv.at<cv::Vec3f>(r, c)[0] < 0.05 || hsv.at<cv::Vec3f>(r, c)[0] > 0.9)
                    if ((hsv.at<cv::Vec3f>(r, c)[1] > 0.3)  && (hsv.at<cv::Vec3f>(r, c)[1] < 0.8) && (hsv.at<cv::Vec3f>(r, c)[2] > 0.2))
                        res.at<uchar>(r, c) = 255;
            }
}

void TextDetector::convertToGrayscale(cv::Mat &gray_img, cv::Mat &input, double ratio) {
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

void TextDetector::applyBinarizationFilter(cv::Mat &bin_img, cv::Mat &gray, int method, double k, double dR) {
    ImageBinarization *img_bin = new ImageBinarization();
    NiblackVersion version = (NiblackVersion) method;
    bin_img = cv::Mat::zeros(gray.size(), gray.type());
    img_bin->applyNiblackSauvolaWolfJolion(gray, bin_img, version, k, dR);
}

void TextDetector::removeUnwantedComponents(cv::Mat &res, cv::Mat &input, int &min, int &max) {
    CBlobResult blobs;
    blobs = CBlobResult(input, cv::Mat(), 4);
    blobs.Filter(blobs, B_INCLUDE, CBlobGetLength(), B_GREATER, min);
    blobs.Filter(blobs, B_EXCLUDE, CBlobGetLength(), B_GREATER, max);
    res = cv::Mat::zeros(input.size(), input.type());
    for (int i = 0; i < blobs.GetNumBlobs(); i++) {
        blobs.GetBlob(i)->FillBlob(res, CV_RGB(255,255,255));
    }
}

void TextDetector::dilateImage(cv::Mat &res, cv::Mat &img, int &h_size) {
    cv::Mat h_structure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(h_size,1));
    cv::dilate(img, res, h_structure);
}

void TextDetector::getMinRectangles(std::vector<cv::RotatedRect> &min_rects, std::vector<std::vector<cv::Point> > &contours, cv::Mat &bin_img) {
    std::vector<cv::Vec4i> hierarchy;
    // find contour
    cv::findContours(bin_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
    for (int i = 0; i < (int)contours.size(); i++) {
        min_rects.push_back(cv::minAreaRect(cv::Mat(contours[i])));
    }
}

// for ID front
void TextDetector::minRectFilter1(cv::Mat &img, cv::Mat &mask, std::vector<cv::RotatedRect> &min_rects) {
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
    }
}

void TextDetector::minRectFilter2(cv::Mat &img, std::vector<cv::RotatedRect> &min_rects) {
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

std::vector<std::string> TextDetector::split(const std::string &s, char delim) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delim))
    {
       tokens.push_back(token);
    }
    return tokens;
}

bool TextDetector::sortFunc(const std::pair<float, float> &A, const std::pair<float, float> &B) {
    return (A.second < B.second);
}

void TextDetector::groupByLine(std::vector<std::vector<cv::RotatedRect> > &mlines, std::vector<cv::RotatedRect> &min_rects, int &h) {
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
            this->sortByX(mindex, line);
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

void TextDetector::groupByLine(std::vector<int> &y_index, std::vector<std::vector<cv::RotatedRect> > &lines, std::vector<cv::RotatedRect> &min_rects, int &h) {
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
        if (line.size() > 0) lines.push_back(line);
    }

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
}

void TextDetector::sortByX(std::vector<int> &indexs, std::vector<cv::RotatedRect> &min_rects) {
//    std::cout << "sort " << min_rects.size() << std::endl;
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

void TextDetector::sortByXY(std::vector<int> &indexs, std::vector<cv::RotatedRect> &min_rects) {

}

void TextDetector::saveParts(std::vector<cv::RotatedRect> &min_rects, cv::Mat &img, std::string &output, std::string &file_name) {
    std::vector<std::string> names = split(file_name,'.');
    std::string m_outpath = output+"/"+names[0];
    mkdir(m_outpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    for (int m = 0; m < (int)min_rects.size(); m++) {

        cv::Rect brect = min_rects[m].boundingRect();
        if (min_rects[m].center == cv::Point2f(-1, -1)) continue;
        float delta = (float)brect.height / DELTA_H_RATIO;
        brect.y -= delta;
        brect.height += (delta+delta);
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
        cv::Mat im_roi = cv::Mat(img, brect).clone();
        std::string roi_name = m_outpath +"/" + names[0] + "_" + std::to_string(m) + ".png";
        cv::imwrite(roi_name, im_roi);
    }
}

void TextDetector::saveParts(std::vector<int> &indexs, std::vector<cv::RotatedRect> &min_rects, cv::Mat &img, std::string &output, std::string &file_name) {
    std::vector<std::string> names = split(file_name,'.');
    std::string m_outpath = output+"/"+names[0];
    mkdir(m_outpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    for (int m = 0; m < (int)indexs.size(); m++) {
        cv::Rect brect = min_rects[indexs[m]].boundingRect();
        if (min_rects[indexs[m]].center == cv::Point2f(-1, -1)) continue;
        float delta = (float)brect.height / DELTA_H_RATIO;
        brect.y -= delta;
        brect.height += (delta+delta);
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
        cv::Mat im_roi = cv::Mat(img, brect).clone();
        std::string roi_name = m_outpath +"/" + names[0] + "_" + std::to_string(m) + ".png";
        cv::imwrite(roi_name, im_roi);
    }
}

void TextDetector::saveParts(std::vector<std::vector<cv::RotatedRect> > &lines, cv::Mat &img, std::string &output, std::string &file_name) {
    std::vector<std::string> names = split(file_name,'.');
    std::string m_outpath = output+"/"+names[0];
    for (int i = 0; i < (int)lines.size(); i++) {
        for (int j = 0; j < (int)lines[i].size(); j++) {
            cv::Rect brect = lines[i][j].boundingRect();
            if (lines[i][j].center == cv::Point2f(-1,-1)) continue;
            float delta = 1;//(float)brect.height / DELTA_H_RATIO;
            brect.y -= delta;
            brect.height += (delta+delta);
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
            cv::Mat im_roi = cv::Mat(img, brect).clone();
            std::string roi_name = m_outpath +"_"+ std::to_string(i)+ "_" + std::to_string(j) + ".png";
            cv::imwrite(roi_name, im_roi);
        }
    }
}

void TextDetector::saveNameField(std::vector<cv::RotatedRect> &name_field, std::vector<std::vector<cv::RotatedRect> > &lines, cv::Mat &img, std::string &output, std::string &file_name) {
    // region filter for name
    int min_x = (float)img.cols / CARD_NAME_X_MIN_RATIO;
    int min_y = (float)img.rows / CARD_NAME_Y_MIN_RATIO;
    int max_x = (float)img.cols / CARD_NAME_X_MAX_RATIO;
    int max_y = (float)img.rows / CARD_NAME_Y_MAX_RATIO;

    name_field.clear();
    std::vector<std::string> names = split(file_name,'.');

    for (int i = 0; i < (int)lines.size(); i++) {
        for (int j = 0; j < (int)lines[i].size(); j++) {
            if (lines[i][j].center == cv::Point2f(-1, -1)) continue;
            if (lines[i][j].center.x > min_x && lines[i][j].center.x < max_x &&
                    lines[i][j].center.y > min_y && lines[i][j].center.y < max_y) {
//                cv::Rect brect = lines[i][j].boundingRect();
//                if (brect.width < brect.height) continue;
//                if (brect.width < 20 && brect.height < 20) continue;
                name_field.push_back(lines[i][j]);
            }
        }
    }
}

void TextDetector::saveDoBField(std::vector<cv::RotatedRect> &dob_field, std::vector<std::vector<cv::RotatedRect> > &lines, cv::Mat &img) {
    // region filter for name
    int min_x = (float)img.cols / CARD_DOB_X_MIN_RATIO;
    int min_y = (float)img.rows / CARD_DOB_Y_MIN_RATIO;
    int max_x = (float)img.cols / CARD_DOB_X_MAX_RATIO;
    int max_y = (float)img.rows / CARD_DOB_Y_MAX_RATIO;

    dob_field.clear();

    for (int i = 0; i < (int)lines.size(); i++) {
        for (int j = 0; j < (int)lines[i].size(); j++) {
            if (lines[i][j].center == cv::Point2f(-1, -1)) continue;
            if (lines[i][j].center.x > min_x && lines[i][j].center.x < max_x &&
                    lines[i][j].center.y > min_y && lines[i][j].center.y < max_y) {
                dob_field.push_back(lines[i][j]);
            }
        }
    }
}

void TextDetector::saveAddrField(std::vector<cv::RotatedRect> &addr_field, std::vector<std::vector<cv::RotatedRect> > &lines, cv::Mat &img) {
    // region filter for name
    int min_x = (float)img.cols / CARD_ADDR_X_MIN_RATIO;
    int min_y = (float)img.rows / CARD_ADDR_Y_MIN_RATIO;
    int max_x = (float)img.cols / CARD_ADDR_X_MAX_RATIO;
    int max_y = (float)img.rows / CARD_ADDR_Y_MAX_RATIO;

    addr_field.clear();

    for (int i = 0; i < (int)lines.size(); i++) {
        for (int j = 0; j < (int)lines[i].size(); j++) {
            if (lines[i][j].center == cv::Point2f(-1, -1)) continue;
            if (lines[i][j].center.x > min_x && lines[i][j].center.x < max_x &&
                    lines[i][j].center.y > min_y && lines[i][j].center.y < max_y) {
                addr_field.push_back(lines[i][j]);
            }

        }
    }

}

void TextDetector::saveIDAreaToImg(cv::Mat &res, cv::Mat &img_bin, cv::Mat &img) {
    cv::Point p1, p2;
    p1.x = (float)img.cols / CARD_ID_X_MIN_RATIO;
    p1.y = (float)img.rows / CARD_ID_Y_MIN_RATIO;
    p2.x = (float)img.cols / CARD_ID_X_MAX_RATIO;
    p2.y = (float)img.rows / CARD_ID_Y_MAX_RATIO;

    cv::Rect roi = cv::Rect(p1, p2);
    cv::Mat color_roi = img(roi);
    cv::Mat img_bin_roi = img_bin(roi);

    this->colorFilter(res, color_roi, RED);
    for (int y = 0; y < (int)res.rows; y++) {
        for (int x = 0; x < (int)res.cols; x++) {
            if (res.at<uchar>(y,x) == 0 && img_bin_roi.at<uchar>(y,x) != 0) {
                res.at<uchar>(y,x) = img_bin_roi.at<uchar>(y,x);
            }
        }
    }
//    cv::imwrite("id_area.png", res);
}

void TextDetector::saveDoBAreaToImg(cv::Mat &res, std::vector<cv::RotatedRect> &min_rects, cv::Mat &img) {
//    for (size_t i = 0; i < min_rects.size(); i++) {
//        cv::Rect m_rect = min_rects[i].boundingRect();
//        std::string m_name = "dob_patch_" + std::to_string(i) + ".png";
//        cv::imwrite(m_name, img(m_rect));
//    }

    if (min_rects.size() == 1) {
        cv::Rect roi = min_rects[0].boundingRect();
        res = img(roi);
    } else {
        cv::Point top_left;
        cv::Point bot_right;
        cv::Rect first_rect = min_rects[0].boundingRect();
        cv::Rect last_rect = min_rects[min_rects.size()-1].boundingRect();

        if (first_rect.y > last_rect.y) top_left.y = last_rect.y;
        else top_left.y = first_rect.y;
        top_left.x = first_rect.x;
        if ((first_rect.y + first_rect.height) < (last_rect.y + last_rect.height)) bot_right.y = last_rect.y + last_rect.height;
        else bot_right.y = first_rect.y + first_rect.height;
        bot_right.x = last_rect.x + last_rect.width;

        cv::Rect roi = cv::Rect(top_left, bot_right);
        res = img(roi);
    }
}

void TextDetector::visualizeRotatedRectsAndContours(cv::Mat &res, std::vector<cv::RotatedRect> &min_rects, std::vector<std::vector<cv::Point> > &contours) {
    float min_h = (float) res.rows / MIN_H_RATIO;
    float max_h = (float) res.rows / MAX_H_RATIO;

    cv::Point2f p1, p2;
    // p1 is filter of image region
    p1.x = (float)res.cols * X_IMG_RATIO;
    p1.y = (float)res.rows * Y_IMG_RATIO;

    // p2 is botom region
    p2.y = (float)res.rows * Y_BOT_RATIO;
    cv::Scalar color = cv::Scalar(255,0,255);
    cv::Scalar color2 = cv::Scalar(255,255,0);
    for (int i = 0; i < (int)contours.size(); i++) {
        // contour
//        cv::drawContours(res, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
        // rotated rect
        cv::Point2f rect_points[4];
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

        if (min_rects[i].center == cv::Point2f(-1, -1)) continue;

        min_rects[i].points(rect_points);
        std::string pos_text = "[" + std::to_string((int)min_rects[i].center.x) + ","+ std::to_string((int)min_rects[i].center.y) + "]";
        cv::circle(res, min_rects[i].center,3, color2, 2);

        for (int j = 0; j < 4; j++) {
            cv::line(res, rect_points[j], rect_points[(j+1)%4], color, 2, 8);
        }
        cv::putText(res, pos_text, min_rects[i].center, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0,255,255),2);
    }
}

void TextDetector::visualizeRotatedRectsByLine(cv::Mat &res, std::vector<std::vector<cv::RotatedRect> > &lines) {
    cv::Scalar color;
    cv::Scalar color2 = cv::Scalar(0,255,0);
    cv::Point2f rect_points[4];
    for (int i = 0; i <(int) lines.size(); i++) {
        if (i%2) color = cv::Scalar(255,0,255);
        else color = cv::Scalar(255,0,0);
        for (int j = 0; j <(int)lines[i].size(); j++) {
            lines[i][j].points(rect_points);
//            std::string pos_text = "[" + std::to_string((int)lines[i][j].center.x) + ","+ std::to_string((int)lines[i][j].center.y) + "]";
            cv::circle(res, lines[i][j].center,3, color2, 2);
            for (int j = 0; j < 4; j++) {
                cv::line(res, rect_points[j], rect_points[(j+1)%4], color, 2, 8);
            }
//            cv::putText(res, pos_text, lines[i][j].center, cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0,255,255),2);
        }
    }

    // visualize regions

}

void TextDetector::processImage(cv::Mat &img, cv::Mat &visualized_img, std::vector<cv::Rect> &rects, int type, bool is_used_filter) {
    rects.clear();
    cv::Mat gray;
    this->convertToGrayscale(gray, img);
    cv::Mat bin_img;
    this->applyBinarizationFilter(bin_img, gray, 1, 0.35, 100);
    // compute min_h, max_h, based on input size
    float img_size = (float)img.rows * (float)img.cols;
    int min_size = img_size / MIN_SIZE_RATIO, max_size = img_size/ MAX_SIZE_RATIO;

    cv::Point2f p1, p2;
    // p1 is filter of image region
    p1.x = (float)img.cols * X_IMG_RATIO;
    p1.y = (float)img.rows * Y_IMG_RATIO;

    // p2 is botom region
    p2.y = (float)img.rows * Y_BOT_RATIO;

    cv::Mat filtered_bin_img;
    cv::Mat inv_bin_img = cv::Scalar::all(255) - bin_img;
    this->removeUnwantedComponents(filtered_bin_img, inv_bin_img, min_size, max_size);
    for (int y = 0; y < inv_bin_img.rows; y++) {
        for (int x = 0; x < inv_bin_img.cols; x++) {
            if (inv_bin_img.at<uchar>(y,x) == 0) {
                filtered_bin_img.at<uchar>(y,x) = 0;
            }
        }
    }
    cv::Mat dilated_bin_img;
    int h_size = 5;
    this->dilateImage(dilated_bin_img, filtered_bin_img, h_size);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::RotatedRect> min_rects;
    this->getMinRectangles(min_rects, contours, dilated_bin_img);
    visualized_img = img.clone();

    if (!is_used_filter) {
        std::vector<std::vector<cv::RotatedRect> > lines;
        this->groupByLine(lines, min_rects, img.rows);
        this->visualizeRotatedRectsByLine(visualized_img, lines);
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
        this->getMinRectangles(min_rects2, contours2, full_mask_bin);
        this->minRectFilter2(img, min_rects2);

        if (is_used_filter) {
            std::vector<std::vector<cv::RotatedRect> > lines;
            this->groupByLine(lines, min_rects2, img.rows);
            this->visualizeRotatedRectsByLine(visualized_img, lines);
            for (size_t i = 0; i < lines.size(); i++) {
                for (size_t j = 0; j < lines[i].size(); j++) {
                    cv::Rect rect = lines[i][j].boundingRect();
                    rects.push_back(rect);
                }
            }
        }
    }
}

void TextDetector::saveBoxesToFile(std::string &name_prefix, std::vector<cv::Rect> &rects, cv::Mat &img) {
    std::string text_name = name_prefix + ".txt";
    std::ofstream text_file(text_name.c_str());

    for (size_t i = 0; i < rects.size(); i++) {
        std::string m_box = std::to_string(rects[i].x) + "," + std::to_string(rects[i].y) + " " +
                std::to_string(rects[i].width) + "," + std::to_string(rects[i].height);
        text_file << m_box << std::endl;
        std::string img_name = name_prefix + "_" + std::to_string(i) + ".png";
        cv::Mat img_roi = img(rects[i]);
        cv::imwrite(img_name, img_roi);
    }
    text_file.close();
}
