#define DETECTOR

#ifdef DETECTOR
#include "TextDetector.h"
#else
#include "TextPreprocessing.h"
#endif

int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cout << "./main_save_boxes_to_file <img_file> <prefix_name>" << std::endl;
        exit(1);
    }
#ifdef DETECTOR
    TextDetector *text = new TextDetector();
#else
    TextPreprocessing *text = new TextPreprocessing();
#endif

    cv::Mat img = cv::imread(argv[1]);
    std::string prefix = std::string(argv[2]);
    cv::Mat visualized_img;
    std::vector<cv::Rect> rects;
    text->processImage(img, visualized_img, rects, FRONT, true);
    std::string visualized_file = prefix +"_visualized.png";
    cv::imwrite(visualized_file, visualized_img);
    text->saveBoxesToFile(prefix, rects, img);
    return 0;


    /**
      Output
      <prefix>.txt
      |- x0,y0 height0,width0 >>> <prefix>_0.png
      |- ....
     **/
}
