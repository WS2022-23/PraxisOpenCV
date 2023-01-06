#include <iostream>
#include <opencv2/opencv.hpp>
#include <config.hpp>
#include "stdio.h"
#include <chrono>
#include <list>

#define WindowName "Test"

using namespace cv;
using namespace std;
using namespace std::chrono;

std::vector<cv::Mat> readVideo(string videoPath) {
    bool ok;
    
    std::vector<cv::Mat> frameVector;
    VideoCapture video;
    video.open(videoPath);
    
    while (video.isOpened()) {
        Mat frame;
        Mat frameCopy;
        ok = video.read(frame);

        if (!ok) {
            break;
        }

        cvtColor(frame, frameCopy, COLOR_RGB2RGBA);
        frameVector.push_back(frameCopy);
        
    }
    
    video.release();
    return frameVector;
}

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
            if (!(imgFront.at<Vec4b>(j, i)[3] <=250 || (imgFront.at<Vec4b>(j, i)[0] == 255 && imgFront.at<Vec4b>(j, i)[1] == 255 && imgFront.at<Vec4b>(j, i)[2] == 255)))
            {
                if((j + pos.y) < imgBack.rows && (j + pos.y) >=0 && (i + pos.x) < imgBack.cols && (i + pos.x) >=0 )
                    imgBack.at<Vec4b>(j + pos.y, i + pos.x) = imgFront.at<Vec4b>(j, i);
            }
        }
    }

    return imgBack;
}


int main(int, char**) {
    
    time_point<steady_clock> begin_time = steady_clock::now(), new_time;
    time_point<steady_clock> snoopBeginTime = steady_clock::now(), snoopTime;
    time_point<steady_clock> jonnyBeginTime = steady_clock::now(), jonnyTime;

    size_t frame_counter = 0;
    size_t fps = 0;
    string openCvPath = OPENCV_PATH;
    openCvPath = openCvPath.substr(0, openCvPath.length() - 12);
    string cascPath = SRC_PATH  "\\haarcascades\\lbpcascade_frontalface_improved.xml";
    string eyeCascPath = SRC_PATH  "\\\\haarcascades\\haarcascade_eye.xml";
    string mouthCascPath = SRC_PATH"\\haarcascades\\haarcascade_mcs_mouth.xml";
    string shoulderCascPath = SRC_PATH"\\haarcascades\\haarcascade_upperbody.xml";
    auto faceCascade = CascadeClassifier(cascPath);
    auto eyeCascade = CascadeClassifier(eyeCascPath);
    auto mouthCascade = CascadeClassifier(mouthCascPath);
    auto shoulderCascade = CascadeClassifier(shoulderCascPath);
    Mat camera_frame;
    Mat origImage;
    Mat dealWithIt = imread(SRC_PATH"\\pictures\\Thug-Life-Sunglasses-PNG.png",IMREAD_UNCHANGED);
    //Mat jonny = imread(SRC_PATH"\\pictures\\joint.png", IMREAD_UNCHANGED);
    string jonnyVideoPath = SRC_PATH"\\pictures\\joint-animated.gif";

    Mat imageGray;

    uint32_t gifIndexSnoop = 0;
    uint32_t gifIndexJonny = 0;

    int camIndex;
    std::list<int> indexList;
    const unsigned char ae = static_cast<unsigned char>(132);
    std::vector<Mat> videoFramesSnoop;
    std::vector<Mat> videoFramesJonny;
    string videoPath = SRC_PATH"\\pictures\\DOGG.gif";
    Mat testFrame;

    std::vector<Rect> faces;
    std::vector<Rect> shoulders;
    VideoCapture cap;

    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    for (int i = -10; i < 11; i++)  {
        cap.open(i);
        if (cap.isOpened())
        {
            indexList.push_back(i);
        }
    }

    std::cout << indexList.size() << " Kamera(s) wurden mit folgenden Indizes gefunden:" << endl;
    for (int i : indexList)
    {
        std::cout << i << endl;
    }
    std::cout << "Index w" << ae << "hlen: ";
    std::cin >> camIndex;

    namedWindow(WindowName, cv::WINDOW_NORMAL);

    cap.open(camIndex);

    if (!cap.isOpened())
    {
        std::cout << "Die Kamera ist geschlossen!!!";
    }

    videoFramesSnoop = readVideo(videoPath);
    videoFramesJonny = readVideo(jonnyVideoPath);

    for(int i = 0; i < videoFramesJonny.size(); i++)
    {
        flip(videoFramesJonny[i], videoFramesJonny[i], 1);
    }

    bool running = true;
    while (running)
    {
        cap >> origImage;
        cvtColor(origImage, camera_frame, COLOR_BGR2BGRA);
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
            //rectangle(camera_frame, face, Scalar(0, 255, 0, 255), 2);
            Mat faceRoi = camera_frame(face);
            vector<Rect> eyes;
            vector<Rect> mouth;
            Mat eyeRoi = faceRoi(Rect(0, 0, faceRoi.cols, faceRoi.rows / 2));
            Mat mouthRoi = faceRoi(Rect(0, faceRoi.rows / 2, faceRoi.cols, faceRoi.rows / 2));
            eyeCascade.detectMultiScale(
                eyeRoi,
                eyes,
                1.1, 2,
                0 | CASCADE_SCALE_IMAGE,
                Size(5, 5)
            );

            mouthCascade.detectMultiScale(
                mouthRoi,
                mouth,
                1.1, 2,
                0 | CASCADE_SCALE_IMAGE,
                Size(5, 5)
            );

            if (eyes.size() >= 2) {
                overlay = Point(face.x + (eyes[0].x + eyes[1].x) / 2.0 + eyes[0].width * 0.5, face.y + (eyes[0].y + eyes[1].y) / 2.0 + eyes[0].height * 0.5);
                Rect roi(overlay.x, overlay.y, face.width, face.height);
                camera_frame = overlayPNG(camera_frame, dealWithIt, roi, true);
                for (int i = 0; i < 2; i++) {
                    center = Point(face.x + eyes[i].x + eyes[i].width * 0.5, face.y + eyes[i].y + eyes[i].height * 0.5);

                    int radius = cvRound((eyes[i].width + eyes[i].height) * 0.25);
                    //circle(camera_frame, center, radius, Scalar(255, 0, 0), 1, 8, 0);
                }
            }
            
            if (mouth.size() >= 1) {
                overlay = Point(face.x + (mouth[0].x + 2 + mouth[0].width / 2), face.y + face.height / 2+ (mouth[0].y + mouth[0].height / 2));
                cv::Rect roi(overlay.x, overlay.y, face.width, face.height);
                //rectangle(camera_frame, roi, Scalar(0, 255, 0, 255), 2);
                if (gifIndexJonny >= videoFramesJonny.size()) {
                    gifIndexJonny = 0;
                }
                jonnyTime = steady_clock::now();

                overlayPNG(camera_frame, videoFramesJonny[gifIndexJonny], roi, false);
                if (jonnyTime - jonnyBeginTime >= milliseconds{ 100 }) {
                    gifIndexJonny++;
                    jonnyBeginTime = jonnyTime;
                }
                
            }
                
        }

        shoulderCascade.detectMultiScale(
            imageGray,
            shoulders,
            1.1,
            3,
            0,
            Size(30, 30)
        );
        for (Rect shoulder : shoulders) {
            //rectangle(camera_frame, shoulder, Scalar(0, 255, 0, 255), 2);
            overlay = Point(shoulder.x+50, shoulder.y + shoulder.height / 2);
            Rect roi(overlay.x, overlay.y, 150, 150);
            if (gifIndexSnoop >= videoFramesSnoop.size()) {
                gifIndexSnoop = 0;
            }
            testFrame = videoFramesSnoop[gifIndexSnoop];
            snoopTime = steady_clock::now();
            camera_frame = overlayPNG(camera_frame, testFrame, roi, true);
            if (snoopTime - snoopBeginTime >= milliseconds{ 100 }) {
                gifIndexSnoop++;
                snoopBeginTime = snoopTime;
            }
            
        }

        frame_counter++;
        new_time = steady_clock::now();
        if (new_time - begin_time >= seconds{ 1 }) {  
            fps = frame_counter;
            frame_counter = 0;
            begin_time = new_time;
        }
        putText(camera_frame, to_string(fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2, LINE_AA);

        imshow(WindowName, camera_frame);
        

        char key = waitKey(1);
        if (key == 27) 
            break;
        running = getWindowProperty(WindowName, WND_PROP_VISIBLE) > 0;
        if (running)
        {
            resizeImage(camera_frame);
        }
    }
    return 0;
}
