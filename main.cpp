#include <iostream>
#include <opencv2/opencv.hpp>
#include <config.hpp>
using namespace cv;
using namespace std;

int main(int, char**) {

    string openCvPath = OPENCV_PATH;
    openCvPath = openCvPath.substr(0, openCvPath.length() - 12);
    string cascPath = openCvPath+ "\\etc\\lbpcascades\\lbpcascade_frontalface_improved.xml";
    auto faceCascade = CascadeClassifier(cascPath);
    
    Mat image;
    Mat imageGray;
    std::vector<Rect> faces;
    namedWindow("Meine Kamera", WINDOW_AUTOSIZE);
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Die Kamera ist geschlossen!!!";
    }
    while (true)
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
        imshow("Display window", image);
        waitKey(25);
    }
    return 0;
}
