#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "config.hpp"
#include <stdio.h>
#include <chrono>
#include <cmath>
#include <string>
#include <iostream>
#include <string>
#include <memory>
#include <atomic>
#include <cmath>
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/deps/status_macros.h"

#define WindowName "Snapchat Filter"
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "landmarks";

using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace mediapipe;

enum handGesture{
    NOTHING,
    THUMB_UP,
    PEACE_SIGN,
    LITTLE_DEVIL,
    OKAY,
    MIDDLE_FINGER,
    FIST
};

// die Zwei Funktionen (get_Euclidean_DistanceAB, detectSign) kamen aus dem Repo https://gist.github.com/TheJLifeX/74958cc59db477a91837244ff598ef4a
// und bestimmen die entfernung der beiden Punkte zueinander und ob die sich auch beführen
float get_Euclidean_DistanceAB(float a_x, float a_y, float b_x, float b_y)
{
    float dist = std::pow(a_x - b_x, 2) + pow(a_y - b_y, 2);
    return std::sqrt(dist);
}
bool isNearFinger(NormalizedLandmark point1, NormalizedLandmark point2)
{
    float distance = get_Euclidean_DistanceAB(point1.x(), point1.y(), point2.x(), point2.y());
    return distance < 0.05;
}
// detectSign wurde hatte seine Inspiration aus https://gist.github.com/TheJLifeX/74958cc59db477a91837244ff598ef4a
// welche beschreibt, wie mann gewisse Posen erkennen kann
// da jedoch die erkennung ob die Finger offen oder geschlossen werden icht richtig funktioniert hatte
// hatten wir uns da was eigenes Überlegt  
int detectSign(mediapipe::NormalizedLandmarkList landmarkList){
    bool thumbIsOpen = false;
    bool firstFingerIsOpen = false;
    bool secondFingerIsOpen = false;
    bool thirdFingerIsOpen = false;
    bool fourthFingerIsOpen = false;
    //

    float pseudoFixKeyPointX0 = (landmarkList.landmark(10).x() - landmarkList.landmark(9).x()) / 2 + landmarkList.landmark(9).x();
    float pseudoFixKeyPointY0 = (landmarkList.landmark(10).y() - landmarkList.landmark(9).y()) / 2 + landmarkList.landmark(9).y();
    if (get_Euclidean_DistanceAB(pseudoFixKeyPointX0,pseudoFixKeyPointY0,landmarkList.landmark(3).x(),landmarkList.landmark(3).y()) 
            < get_Euclidean_DistanceAB(pseudoFixKeyPointX0,pseudoFixKeyPointY0,landmarkList.landmark(4).x(),landmarkList.landmark(4).y()))
    {
        thumbIsOpen = true;
    }

    float pseudoFixKeyPointX = landmarkList.landmark(0).x();
    float pseudoFixKeyPointY = landmarkList.landmark(0).y();
    if (get_Euclidean_DistanceAB(pseudoFixKeyPointX,pseudoFixKeyPointY,landmarkList.landmark(7).x(),landmarkList.landmark(7).y()) 
            < get_Euclidean_DistanceAB(pseudoFixKeyPointX,pseudoFixKeyPointY,landmarkList.landmark(8).x(),landmarkList.landmark(8).y()))
    {
        firstFingerIsOpen = true;
    }

    if (get_Euclidean_DistanceAB(pseudoFixKeyPointX,pseudoFixKeyPointY,landmarkList.landmark(10).x(),landmarkList.landmark(10).y()) 
            < get_Euclidean_DistanceAB(pseudoFixKeyPointX,pseudoFixKeyPointY,landmarkList.landmark(12).x(),landmarkList.landmark(12).y()))
    {
        secondFingerIsOpen = true;
    }

    if (get_Euclidean_DistanceAB(pseudoFixKeyPointX,pseudoFixKeyPointY,landmarkList.landmark(14).x(),landmarkList.landmark(14).y()) 
            < get_Euclidean_DistanceAB(pseudoFixKeyPointX,pseudoFixKeyPointY,landmarkList.landmark(16).x(),landmarkList.landmark(16).y()))
    {
        thirdFingerIsOpen = true;
    }
    
    if (get_Euclidean_DistanceAB(pseudoFixKeyPointX,pseudoFixKeyPointY,landmarkList.landmark(18).x(),landmarkList.landmark(18).y()) 
            < get_Euclidean_DistanceAB(pseudoFixKeyPointX,pseudoFixKeyPointY,landmarkList.landmark(20).x(),landmarkList.landmark(20).y()))
    {
        fourthFingerIsOpen = true;
    }

    // Hand gesture recognition
    if (thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        return THUMB_UP;
    }
    else if (!thumbIsOpen && firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        return PEACE_SIGN;
    }
    else if (!thumbIsOpen && firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && fourthFingerIsOpen)
    {
        return LITTLE_DEVIL;
    }
    else if (!firstFingerIsOpen && secondFingerIsOpen && thirdFingerIsOpen && fourthFingerIsOpen && isNearFinger(landmarkList.landmark(4), landmarkList.landmark(8)))
    {
        return OKAY;
    }
    else if (!firstFingerIsOpen && secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        return MIDDLE_FINGER;
    }
    else if (!thumbIsOpen && !firstFingerIsOpen && !secondFingerIsOpen && !thirdFingerIsOpen && !fourthFingerIsOpen)
    {
        return FIST;
    }
    return -1;
}

// nimmt von einem GIF oder einer Video-Datei jeden Frame und speichert diese in einen Vektor 
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

    cv::Rect windowSizes = getWindowImageRect(WindowName);

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

void overlayPNG(Mat& imgBack, Mat imgFront, Rect pos = Rect(), bool centered = false, bool gif = false) {

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
            if (gif) {
                if (!(imgFront.at<Vec4b>(j, i)[0] == 255 && imgFront.at<Vec4b>(j, i)[1] == 255 && imgFront.at<Vec4b>(j, i)[2] == 255))
                {
                    if ((j + pos.y) < imgBack.rows && (j + pos.y) >= 0 && (i + pos.x) < imgBack.cols && (i + pos.x) >= 0)
                        imgBack.at<Vec4b>(j + pos.y, i + pos.x) = imgFront.at<Vec4b>(j, i);
                }
            }
            else {
                if (!(imgFront.at<Vec4b>(j, i)[3] <=250))
            {
                    if((j + pos.y) < imgBack.rows && (j + pos.y) >=0 && (i + pos.x) < imgBack.cols && (i + pos.x) >=0 )
                        imgBack.at<Vec4b>(j + pos.y, i + pos.x) = imgFront.at<Vec4b>(j, i);
                }
            }
            
        }
    }
}

mediapipe::Status run(){
    // Uhren für verschiedene Aktionen
    //Frames per Seconds anzeigen
    time_point<steady_clock> begin_time = steady_clock::now();
    time_point<steady_clock> new_time;
    size_t frame_counter = 0;
    size_t fps = 0;
    // Uhr um zu bestimmen ob gezeichnet werden soll oder nicht
    time_point<steady_clock> begin_time_drawing = steady_clock::now();
    time_point<steady_clock> new_time_drawing;
    bool drawing = false;
    // Uhr um zu bestimmen ob auf die Frames eine Kanten erkennung gemacht werden soll
    time_point<steady_clock> begin_time_Edge = steady_clock::now();
    time_point<steady_clock> new_time_Edge;
    bool inverse = false;
    bool drawing_Edge = false;
    // Uhr um zu bestimmen ob alle gezeichneten Linien gelöscht werden sollen
    time_point<steady_clock> begin_time_CLEAR = steady_clock::now();
    time_point<steady_clock> new_time_CLEAR;
    bool clear = false;
    //Zeit für die Animationen
    time_point<steady_clock> snoopBeginTime = steady_clock::now(), snoopTime;
    time_point<steady_clock> jonnyBeginTime = steady_clock::now(), jonnyTime;
    uint32_t gifIndexSnoop = 0;
    uint32_t gifIndexJonny = 0;

    //Variablen für die Hände
    mediapipe::NormalizedLandmarkList single_hand_NormalizedLandmarkList;
    cv::Rect Hand;
    int handWidth = 0;
    int handHeight = 0;
    std::vector<int> hand_Breite;
    std::vector<int> hand_Hoehe;
    std::vector<cv::Point> hand_landmarks;
    std::vector<int> hand_landmarks_lines;
    std::vector<Scalar> hand_landmarks_Color;
    cv::Point fingertip;    
    bool skip = false;     

    // OpenCV: Pfade zu Erkennungsmodelle
    string openCvPath = OPENCV_PATH;
    openCvPath = openCvPath.substr(0, openCvPath.length() - 12);
    string cascPath = SRC_PATH  "\\haarcascades\\lbpcascade_frontalface_improved.xml";
    string eyeCascPath = SRC_PATH  "\\haarcascades\\haarcascade_eye.xml";
    string mouthCascPath = SRC_PATH"\\haarcascades\\haarcascade_mcs_mouth.xml";
    string shoulderCascPath = SRC_PATH"\\haarcascades\\haarcascade_upperbody.xml";
    auto faceCascade = CascadeClassifier(cascPath);
    std::vector<cv::Rect> faces;
    std::vector<cv::Rect> shoulders;
    auto eyeCascade = CascadeClassifier(eyeCascPath);
    auto mouthCascade = CascadeClassifier(mouthCascPath);
    auto shoulderCascade = CascadeClassifier(shoulderCascPath);

    // Einlesen von Bildern
    cv::Mat dealWithIt = imread(SRC_PATH"\\pictures\\Thug-Life-Sunglasses-PNG.png",IMREAD_UNCHANGED);
    //cv::Mat jonny = imread(SRC_PATH"\\pictures\\joint.png", IMREAD_UNCHANGED);
    cv::Mat middleFinger = imread(SRC_PATH"\\pictures\\MiddleFinger.png",IMREAD_UNCHANGED);
    cv::Mat pepeOK = imread(SRC_PATH"\\pictures\\pepeOK.png",IMREAD_UNCHANGED);
    cv::Mat peace = imread(SRC_PATH"\\pictures\\peace.png",IMREAD_UNCHANGED);
    cv::Mat devil = imread(SRC_PATH"\\pictures\\Angry_Devil_Emoji_large.png",IMREAD_UNCHANGED);
    cv::Mat theRock = imread(SRC_PATH"\\pictures\\theRock.png",IMREAD_UNCHANGED);
    cv::Mat OkayMeme = imread(SRC_PATH"\\pictures\\OkayMeme.png",IMREAD_UNCHANGED);
    string jonnyVideoPath = SRC_PATH"\\pictures\\joint-animated.gif";
    
    mediapipe::Packet packet;
    int landmarkmaxsize = 600;
    Scalar color(0, 255, 0, 255);
    double res = 100.0f;
    int sign = -1;

    // einlesen des Calculator zum erkennen von Händen
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        "mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt",
        &calculator_graph_config_contents));
    // LOG(INFO) << "Get calculator graph config contents: "
    //         << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));
    LOG(INFO) << "finished.";

    // Variablen für die Kamera
    int camIndex;
    std::list<int> indexList;
    const unsigned char ae = static_cast<unsigned char>(132);
    std::vector<Mat> videoFramesSnoop;
    std::vector<Mat> videoFramesJonny;
    string videoPath = SRC_PATH"\\pictures\\DOGG.gif";
    Mat testFrame;
    VideoCapture cap;
    
    for (int i = -10; i < 11; i++)  {
        cap.open(i);
        if (cap.isOpened())
        {
            indexList.push_back(i);
        }
    }
    //Auswahl aller Kameras
    std::cout << indexList.size() << " Kamera(s) wurden mit folgenden Indizes gefunden:" << endl;
    for (int i : indexList)
    {
        std::cout << i << endl;
    }
    std::cout << "Index w" << ae << "hlen: ";
    std::cin >> camIndex;

    //Deaktivieren der Bilder
    bool run = true;
    int Index = -1; 
    std::cout << "0: Deaktivieren der Bilder" << endl;
    std::cout << "1: Aktivieren der Bilder" << endl;
    while (!(Index == 1 || Index == 0))
    {
        std::cin >> Index;
    }
    run = bool(Index);

    // erstellt benutztes Fenster in dem alles gerendert wird
    namedWindow(WindowName, cv::WINDOW_NORMAL);

    cap.open(camIndex);
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 60);

    if (!cap.isOpened())
        std::cout << "Die Kamera ist geschlossen!!!\n";
    float distance = 0.04f * cap.get(CAP_PROP_FRAME_HEIGHT);

    videoFramesSnoop = readVideo(videoPath);
    videoFramesJonny = readVideo(jonnyVideoPath);

    for(int i = 0; i < videoFramesJonny.size(); i++)
    {
        flip(videoFramesJonny[i], videoFramesJonny[i], 1);
    }

    // Versucht die Graphen zu benutzen
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                    graph.AddOutputStreamPoller(kOutputStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    bool running = true;
    while (running)
    {        
        cv::Mat camera_frame_raw;
        cap >> camera_frame_raw;
        if (camera_frame_raw.empty()) {
            LOG(INFO) << "Empty frame, end of video reached.";
            break;
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);

        // Wandelt OpenCV Mat in ein Format welches Mediapipe auch benutzen kann
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);
    
        // Sendet das geholte Kamerabild an den Graphen
        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));

        // Falls eine Hand gefunden wird sollen die ganzen Aktionen mit der Hand ausgeführt werden
        if(poller.QueueSize() > 0){
            if (!poller.Next(&packet)) break;
            // Holt sich alle Punkte der gefundenen Hand
            auto& output_landmark = packet.Get<std::vector<class mediapipe::NormalizedLandmarkList>>();

            for (int m = 0; m < output_landmark.size(); ++m){
                // alle 21 Punkte der Hand werden hier eingespeichert
                single_hand_NormalizedLandmarkList = output_landmark[m];
                // soll zwischen sechs verschiedenen Handzeichen unterscheiden, falls keins gefunden wird der ausgabewert -1 sein
                sign = detectSign(single_hand_NormalizedLandmarkList);
                const mediapipe::NormalizedLandmark landmark_Thumb = single_hand_NormalizedLandmarkList.landmark(4);
                const mediapipe::NormalizedLandmark landmark_Pinky = single_hand_NormalizedLandmarkList.landmark(20);
                const mediapipe::NormalizedLandmark landmark_Middle = single_hand_NormalizedLandmarkList.landmark(12);
                const mediapipe::NormalizedLandmark landmark_Ring = single_hand_NormalizedLandmarkList.landmark(16);
                // Holt sich die Distanz zwischen den jeweiligen Punkte
                res = cv::norm(cv::Point(landmark_Thumb.x() * camera_frame.cols,landmark_Thumb.y() * camera_frame.rows)-cv::Point(landmark_Pinky.x() * camera_frame.cols,landmark_Pinky.y() * camera_frame.rows)); 
                inverse = (cv::norm(cv::Point(landmark_Thumb.x() * camera_frame.cols,landmark_Thumb.y() * camera_frame.rows)-cv::Point(landmark_Middle.x() * camera_frame.cols,landmark_Middle.y() * camera_frame.rows))) <= distance;
                clear = (cv::norm(cv::Point(landmark_Thumb.x() * camera_frame.cols,landmark_Thumb.y() * camera_frame.rows)-cv::Point(landmark_Ring.x() * camera_frame.cols,landmark_Ring.y() * camera_frame.rows))) <= distance;
                
                // Wenn sich die Finger nicht berühren sollen die ganzen Uhren immer das gleiche sein
                new_time_drawing = steady_clock::now();
                if(res > distance  && sign == -1){
                    begin_time_drawing = new_time_drawing;
                }
                new_time_Edge = steady_clock::now();
                if(!inverse  && sign == -1){
                    begin_time_Edge = new_time_Edge;
                }
                new_time_CLEAR = steady_clock::now();
                if(!clear  && sign == -1){
                    begin_time_CLEAR = new_time_CLEAR;
                }

                // Zum zeichnen der Handpunkte und einspeisen der nächsten Position damit dann gezeichnet werden kann
                for (int i = 0; i < single_hand_NormalizedLandmarkList.landmark_size(); ++i){
                    const mediapipe::NormalizedLandmark landmark = single_hand_NormalizedLandmarkList.landmark(i);
                    int x = landmark.x() * camera_frame.cols;
                    int y = landmark.y() * camera_frame.rows;
                    fingertip = cv::Point(x,y);
                    circle(camera_frame, fingertip, 3, Scalar(255,0,0),-1, 8, 0);
                    if (drawing && sign == -1)
                    {
                        // es soll nur beim ZeigeFinger gezeichnet werden deswegen das i == 8
                        if(i == 8 && res > distance) {
                            hand_landmarks.push_back(fingertip);
                            // wenn sich die Farbe ändert
                            hand_landmarks_Color.push_back(color);
                            // wenn man Pausiert das da dann auch eine Lücke zwischen den Punkten bleiben
                            if(!skip){
                                if(hand_landmarks_lines.size() == 0) {
                                    if (hand_landmarks.size() == 2)
                                    {
                                        hand_landmarks_lines.push_back(1);
                                    }
                                }
                                else {
                                    hand_landmarks_lines.push_back(1);
                                }
                            } 
                            else {
                                if(hand_landmarks_lines.size() > 0) {
                                    hand_landmarks_lines.push_back(0);
                                }
                                skip = false;
                            }
                            // wenn zu viele Punkte sind werden die ersten immer gelöscht
                            if (hand_landmarks.size() > landmarkmaxsize)
                            {
                                hand_landmarks.erase(hand_landmarks.begin());
                            }
                            if (hand_landmarks_Color.size() > landmarkmaxsize)
                            {
                                hand_landmarks_Color.erase(hand_landmarks_Color.begin());
                            }
                            if (hand_landmarks_lines.size() > landmarkmaxsize-1)
                            {
                                hand_landmarks_lines.erase(hand_landmarks_lines.begin());
                            }
                        }
                        else if(i == 8 && res <= distance)
                        {
                            skip = true;
                        }  
                    }
                    hand_Breite.push_back(x);   
                    hand_Hoehe.push_back(y);
                    
                }
                // wenn sich der kleine Finger und Daumen berühren und das für 2 Sekunden, dann wird das Zeichnen aus oder an gemacht 
                if(res <= distance && sign == -1){ 
                    if (new_time_drawing - begin_time_drawing >= seconds{ 2 }) {  
                        begin_time_drawing = new_time_drawing;
                        drawing = !drawing;
                        if(drawing == false) {
                            skip = true;
                            color = cv::Scalar(
                                (double)std::rand() / RAND_MAX * 255,
                                (double)std::rand() / RAND_MAX * 255,
                                (double)std::rand() / RAND_MAX * 255,
                                255
                            );
                        }
                    }
                }
                // wenn sich Daumen und Mittelfinger berühren wird im Bild die Kanten erkannt 
                if (inverse && sign == -1) {
                    if (new_time_Edge - begin_time_Edge >= seconds{ 2 }) {  
                        begin_time_Edge = new_time_Edge;
                        drawing_Edge = !drawing_Edge;
                    }
                }
                // wenn sich Daumen und Ringfinger berühren werden alle gespeicherten Punkte gelöscht
                if (clear && sign == -1) {
                    if (new_time_CLEAR - begin_time_CLEAR >= seconds{ 2 }) {  
                        begin_time_CLEAR = new_time_CLEAR;
                        hand_landmarks_lines.clear();
                        hand_landmarks.clear();
                        hand_landmarks_Color.clear();
                        color = cv::Scalar(
                            (double)std::rand() / RAND_MAX * 255,
                            (double)std::rand() / RAND_MAX * 255,
                            (double)std::rand() / RAND_MAX * 255,
                            255
                        );
                    }
                }
                
            }

            //Erstelle mir damit einen Kasten wo die Ganze Hand enthalten ist
            std::sort(hand_Breite.begin(),hand_Breite.end());
            std::sort(hand_Hoehe.begin(),hand_Hoehe.end());
            Hand.x = hand_Breite[0] - 10;
            Hand.y = hand_Hoehe[0] - 10;
            Hand.width = (hand_Breite[hand_Breite.size()-1] - Hand.x ) + 10;
            Hand.height = (hand_Hoehe[hand_Breite.size()-1] - Hand.y ) + 10;
        }

        //Gibt wieder ob man sich gerade im Zeichnen Modus oder nicht befindet
        if (drawing)
        {
            cv::putText(camera_frame, "Drawing Activated", Point(30, camera_frame.rows - 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2, LINE_AA);
        }else {
            cv::putText(camera_frame, "Drawing Deactivated", Point(30, camera_frame.rows - 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0, 255), 2, LINE_AA);
        }
        

        cv::cvtColor(camera_frame, camera_frame, cv::COLOR_BGR2RGBA);
        cv::Mat output_frame_mat_grey;
        cv::cvtColor(camera_frame, output_frame_mat_grey, cv::COLOR_BGR2RGBA);
        
            // damit soll mehrere Gesichter erkennen
            faceCascade.detectMultiScale(
                output_frame_mat_grey,
                faces,
                1.1,
                3,
                0,
                Size(30, 30)
            );
            Point center;
            Point overlay;
            // alle erkennten Gesichter bekommen einige Bilder aufgesetzt 
            for (cv::Rect face : faces)
            {
                Mat faceRoi = camera_frame(face);
                vector<cv::Rect> eyes;
                vector<cv::Rect> mouth;
                Mat eyeRoi = faceRoi(Rect(0, 0, faceRoi.cols, faceRoi.rows / 2));
                Mat mouthRoi = faceRoi(Rect(0, faceRoi.rows / 2, faceRoi.cols, faceRoi.rows / 2));

                //Soll Augen erkennen
                eyeCascade.detectMultiScale(
                    eyeRoi,
                    eyes,
                    1.1, 2,
                    0 | CASCADE_SCALE_IMAGE,
                    Size(5, 5)
                );
                //Soll Münder erkennen
                mouthCascade.detectMultiScale(
                    mouthRoi,
                    mouth,
                    1.1, 2,
                    0 | CASCADE_SCALE_IMAGE,
                    Size(5, 5)
                ); 
                // Zeichnet einen Sonnenbrille auf jedes Augenpaar welches es finden kann
                if (eyes.size() >= 2) {
                    overlay = Point(face.x + (eyes[0].x + eyes[1].x) / 2.0 + eyes[0].width * 0.5, face.y + (eyes[0].y + eyes[1].y) / 2.0 + eyes[0].height * 0.5);
                    cv::Rect roi(overlay.x, overlay.y, face.width, face.height);
                    //camera_frame = overlayPNG(camera_frame, dealWithIt, roi, true);
                    overlayPNG(camera_frame, dealWithIt, roi, true);
                    for (int i = 0; i < 2; i++) {
                        center = Point(face.x + eyes[i].x + eyes[i].width * 0.5, face.y + eyes[i].y + eyes[i].height * 0.5);

                        int radius = cvRound((eyes[i].width + eyes[i].height) * 0.25);
                        //circle(camera_frame, center, radius, Scalar(255, 0, 0), 1, 8, 0);
                    }
                }
                // Zeichnet einen Joint auf jeden Mund welches es finden kann
                if (mouth.size() >= 1) {
                    overlay = Point(face.x + (mouth[0].x + 2 + mouth[0].width /2), face.y + face.height/2 + (mouth[0].y + mouth[0].height/2));
                    cv::Rect roi(overlay.x, overlay.y, face.width, face.height);
                    if (gifIndexJonny >= videoFramesJonny.size()) {
                        gifIndexJonny = 0;
                    }
                    jonnyTime = steady_clock::now();

                    overlayPNG(camera_frame, videoFramesJonny[gifIndexJonny], roi, false, true);
                    if (jonnyTime - jonnyBeginTime >= milliseconds{ 100 }) {
                        gifIndexJonny++;
                        jonnyBeginTime = jonnyTime;
                    }
                }
            }
        if(run) {
            shoulderCascade.detectMultiScale(
                output_frame_mat_grey,
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
                overlayPNG(camera_frame, testFrame, roi, true, true);
                if (snoopTime - snoopBeginTime >= milliseconds{ 100 }) {
                    gifIndexSnoop++;
                    snoopBeginTime = snoopTime;
                }
                break;
            }
        }
        
        // Zeichnet die Punkte und Linien, die durch den Zeigefinger erzeugt werden
        for (size_t i = 1; i < hand_landmarks.size(); i++)
        {
            circle(camera_frame, hand_landmarks[i-1], 7, hand_landmarks_Color[i-1],-1, 8, 0);
            if (hand_landmarks_lines.size() >= 1) {
                if (hand_landmarks_lines[i-1] == 1) {
                    line(camera_frame, hand_landmarks[i-1], hand_landmarks[i], hand_landmarks_Color[i-1], 14);
                }
            }
        }
        if(hand_landmarks.size() > 0) {
            circle(camera_frame, hand_landmarks[hand_landmarks.size()-1], 7, hand_landmarks_Color[hand_landmarks_Color.size()-1],-1, 8, 0);
        
        }

        
        // Wenn eines der Switch-Cases ausgewählt wird, werden auf der getrackten Hand das dazugehörige Bild gezeichnet
        switch (sign)
        {
        case THUMB_UP:
            cv::putText(camera_frame, "thumb UP", Point(30, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2, LINE_AA);
            overlayPNG(camera_frame,pepeOK, Hand, false);
            break;
        case PEACE_SIGN:
            cv::putText(camera_frame, "PEACE SIGN", Point(30, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2, LINE_AA);
            overlayPNG(camera_frame,peace, Hand, false);
            break;
        case LITTLE_DEVIL:
            cv::putText(camera_frame, "little Devil", Point(30, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2, LINE_AA);
            overlayPNG(camera_frame,devil, Hand, false);
            break;
        case OKAY:
            cv::putText(camera_frame, "OKAY", Point(30, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2, LINE_AA);
            overlayPNG(camera_frame,OkayMeme, Hand, false);
            break;
        case MIDDLE_FINGER:
            cv::putText(camera_frame, "MIDDLE FINGER", Point(30, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2, LINE_AA);
            overlayPNG(camera_frame, middleFinger, Hand, false);
            break;
        case FIST:
            cv::putText(camera_frame, "FIST", Point(30, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2, LINE_AA);
            overlayPNG(camera_frame, theRock, Hand, false);
            break;
        default:
            break;
        }

        //Macht eine kleine Kanten erkennung mit dem Canny-Verfahren und inversiert die Farbwerte
        if (drawing_Edge)
        {
            cv::Mat canny_edges;
            cv::Canny(camera_frame,canny_edges,100,200,3,false);
            cv::bitwise_not(canny_edges, camera_frame);
        }

        // Bestimmt die Frames welche in einer Sekunde gerendert werden können
        frame_counter++;
        new_time = steady_clock::now();
        if (new_time - begin_time >= seconds{ 1 }) {  
            fps = frame_counter;
            frame_counter = 0;
            begin_time = new_time;
        }
        cv::putText(camera_frame, to_string(fps), Point(30, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0, 255), 2, LINE_AA);

        // Zeigt das Bild auf dem Fenster an
        cv::imshow(WindowName, camera_frame);

        //Die Hand wird hier zurückgesetzt
        Hand = cv::Rect();
        hand_Breite.clear();
        hand_Hoehe.clear();

        // Wartet hier auf das Drücken der ESC-Taste da, somit das Programm beendet wird
        char key = waitKey(1);
        if (key == 27) 
            break;

        // Schaut nach das Fenster geschossen werden soll
        running = getWindowProperty(WindowName, WND_PROP_VISIBLE) > 0;
        // Wenn das Fenster noch am Leben gelassen wird, dann wird das Fenster auf 16 zu 9 gerendet
        if (running)
        {
            resizeImage(camera_frame);
        }
    }
    hand_landmarks.clear();
    hand_landmarks_lines.clear();
    hand_landmarks_Color.clear();
    cv::destroyAllWindows();

    return mediapipe::OkStatus();
}


int main(int, char**) {

    mediapipe::Status status = run();
    if (!status.ok()) {
        LOG(ERROR) << "Failed to run the graph: " << status.message();
        return EXIT_FAILURE;
    } else {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
