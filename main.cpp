#include <iostream>
#include <opencv2/opencv.hpp>
#include <config.hpp>
#include "stdio.h"
#include <chrono>

#define WindowName "Test"

using namespace cv;
using namespace std;
using namespace std::chrono;



void resizeImage(Mat& image) {

    Rect windowSizes = getWindowImageRect(WindowName);

    int newSize = (windowSizes.width * 1.0f / (windowSizes.height));
    if ((16.0f/9) != newSize)
    {
        int temp = (16 * 1.0f / 9) * windowSizes.height;
        resizeWindow(WindowName, Size(temp, windowSizes.height));
    }

    float ratioHeight = (windowSizes.height) / (image.size().height * 1.0f);
    float ratioWidth = (windowSizes.width) / (image.size().width * 1.0f);

    resize(image, image, Size(), ratioWidth, ratioHeight);
}

int main(int, char**) {
    
    time_point<steady_clock> begin_time = steady_clock::now(), new_time;
    size_t frame_counter = 0;
    size_t fps = 0;
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

        frame_counter++;
        new_time = steady_clock::now();
        if (new_time - begin_time >= seconds{ 1 }) {  
            fps = frame_counter;
            frame_counter = 0;
            begin_time = new_time;
        }
        putText(image, to_string(fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2, LINE_AA);

        imshow(WindowName, image);
        

        char key = waitKey(1);
        if (key == 27) 
            break;
        running = getWindowProperty(WindowName, WND_PROP_VISIBLE) > 0;
        if (running)
        {
            resizeImage(image);
        }
    }
    return 0;
}