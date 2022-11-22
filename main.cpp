#include <iostream>
#include <opencv2/opencv.hpp>
#include <config.hpp>
#include "stdio.h"

#define WindowName "Test"

using namespace cv;
using namespace std;



void resizeImage(Mat& image) {
    try
    {
        Rect sizes = getWindowImageRect(WindowName);

        int a = (sizes.width * 1.0f / (sizes.height));
        if ((16.0f/9) != a)
        {
            int temp = (16 * 1.0f / 9) * sizes.height;
            resizeWindow(WindowName, Size(temp, sizes.height));
        }

        float ratioHeight = (sizes.height) / (image.size().height * 1.0f);
        float ratioWidth = (sizes.width) / (image.size().width * 1.0f);

        resize(image, image, Size(), ratioWidth, ratioHeight);
    }
    catch (const std::exception&)
    {
        //exit(-69);
    }


}

int main(int, char**) {

    string openCvPath = OPENCV_PATH;
    openCvPath = openCvPath.substr(0, openCvPath.length() - 12);
    string cascPath = openCvPath+ "\\etc\\lbpcascades\\lbpcascade_frontalface_improved.xml";
    auto faceCascade = CascadeClassifier(cascPath);
    
    Mat image;
    Mat imageGray;
    std::vector<Rect> faces;
    namedWindow(WindowName, cv::WINDOW_NORMAL);
    VideoCapture cap(0);

    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    if (!cap.isOpened())
    {
        std::cout << "Die Kamera ist geschlossen!!!";
    }
    bool running = true;
    while (running)
    {
        cap >> image;

        cvtColor(image, imageGray, COLOR_BGR2GRAY);
        faceCascade.detectMultiScale(
            imageGray,
            faces,
            1.1,
            3,
            0,
            Size(30, 30)
        );
        for (Rect face : faces)
        {
            rectangle(image, face, Scalar(0, 255, 0, 255), 2);
        }

        resizeImage(image);
        
        imshow(WindowName, image);
        

        char key = waitKey(70);
        if (key == 27) 
            break;
        running = getWindowProperty(WindowName, WND_PROP_VISIBLE) > 0;
    }
    return 0;
}