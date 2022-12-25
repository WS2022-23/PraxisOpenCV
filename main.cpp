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


Mat overlayPNG(Mat imgBack, Mat imgFront, Rect pos = Rect(), bool centered = false) {

    if (pos == Rect()) {
        pos = Rect(0, 0, imgBack.cols, imgBack.rows);
    }

    
    //pos = Rect(0, 0, imgFront.cols , imgFront.rows-100);
    Size sizePos(min(pos.width,imgBack.cols), min(pos.height, imgBack.rows));
    float aspectPos = sizePos.width / (float)sizePos.height;
    float aspectFront = imgFront.cols / (float)imgFront.rows;


    if (imgFront.cols > pos.width) {
        resize(imgFront, imgFront, Size(pos.width, round(pos.width / aspectFront)));
    }
    if (imgFront.rows > pos.height) {
        resize(imgFront, imgFront, Size(round(pos.height * aspectFront),pos.height ));
    }

    if (centered) {
        pos.x -= imgFront.cols / 2;
        pos.y -= imgFront.rows / 2;
    }
    for (int i = 0; i < imgFront.cols; i++)
    {
        for (int j = 0; j < imgFront.rows; j++)
        {
            if (!(imgFront.at<Vec4b>(j, i)[3] <=250 ))
            {
                if((j + pos.y) < imgBack.rows && (i + pos.x) < imgBack.cols )
                    imgBack.at<Vec4b>(j + pos.y, i + pos.x) = imgFront.at<Vec4b>(j, i);
            }
        }
    }

    return imgBack;
}


int main(int, char**) {
    
    time_point<steady_clock> begin_time = steady_clock::now(), new_time;
    size_t frame_counter = 0;
    size_t fps = 0;
    string openCvPath = OPENCV_PATH;
    openCvPath = openCvPath.substr(0, openCvPath.length() - 12);
    string cascPath = openCvPath+ "\\etc\\lbpcascades\\lbpcascade_frontalface_improved.xml";
    string eyeCascPath = openCvPath + "\\etc\\haarcascades\\haarcascade_eye.xml";

    auto faceCascade = CascadeClassifier(cascPath);
    auto eyeCascade = CascadeClassifier(eyeCascPath);
    Mat image;
    Mat origImage;
    Mat dealWithIt = imread(SRC_PATH"\\Thug-Life-Sunglasses-PNG.png",IMREAD_UNCHANGED);
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
        cap >> origImage;
        cvtColor(origImage, image, COLOR_BGR2BGRA);
        cvtColor(origImage, imageGray, COLOR_BGR2GRAY);
        faceCascade.detectMultiScale(
            imageGray,
            faces,
            1.1,
            3,
            0,
            Size(30, 30)
        );
        Point center;
        Point overlay;
        
        for (Rect face : faces)
        {
            rectangle(image, face, Scalar(0, 255, 0, 255), 2);
            Mat faceRoi = image(face);
            vector<Rect> eyes;

            eyeCascade.detectMultiScale(
                faceRoi,
                eyes,
                1.1, 2,
                0 | CASCADE_SCALE_IMAGE,
                Size(5, 5)
            );
            if (eyes.size() >= 2) {
                overlay = Point(face.x + (eyes[0].x + eyes[1].x) / 2.0 + eyes[0].width * 0.5, face.y + (eyes[0].y + eyes[1].y) / 2.0 + eyes[0].height * 0.5);
                Rect roi(overlay.x, overlay.y, face.width, face.height);
                image = overlayPNG(image, dealWithIt, roi, true);
                for (int i = 0; i < 2; i++) {
                    center = Point(face.x + eyes[i].x + eyes[i].width * 0.5, face.y + eyes[i].y + eyes[i].height * 0.5);

                    int radius = cvRound((eyes[i].width + eyes[i].height) * 0.25);
                    circle(image, center, radius, Scalar(255, 0, 0), 1, 8, 0);
                }
            }
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
