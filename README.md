# BildVerarbeitung_OpenCV
In diesem Repository befindet sich das Praktikumsprojekt aus der Veranstaltung Bildverarbeitung 1 im WiSe 2022/23
die main.cpp ist lediglich eine Kopie und kann nicht durch zu einer Ausführbare umgewandelt werden


## Projekt mit Microsoft Visual Studio starten
    - Benötigt:
        - Cmake (welches im Pfad hinzugefügt wurde)
        - Microsoft Visual Studio

    - Ablauf:
        - Cmake öffnen
        - Den Ordner des Repos auswählen
        - Den Build Ordner außerhalb des Repos erstellen
        - Um das Projekt in Visual Studio bearbeiten:
            - die unteren Knöpfe einmal nach der Reihe drücken
                - Configure
                - Generate
                - Open Project

# Anleitung für Mediapipe und Bazel etc.
    Einfach so machen wie es da beschrieben ist
        https://ritvik-mandyam.medium.com/mediapipe-on-windows-a-tale-of-woe-4c4d848b4dab
        und
        https://google.github.io/mediapipe/getting_started/install.html#installing-on-windows (für MSYS2)

    Benötigt werden:
        Python 3.8.7
        Visual C++ Build Tools für 2019
        MSYS2
        bazel 5.2.0

    Bei WinSDK vorher schauen ob es schon welche gibt und diese dann löschen (so hab ich des zumindestens gemacht)
    

## Bauen und Ausführen des Programmes

    Bevor es mit dem Bauen losgehen kann muss das Projekt erstmal mit Cmake gebaut werden
    Nachdem das Bauen erfolgreich war muss die config.h Datei in den Ordner "mediapipe/mediapipe/examples/desktop/BivaProject" 
        ersetzt werden, da hier Pfade dabei sind

    cd mediapipe
    set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
    set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C:/development/python.exe" mediapipe/examples/desktop/BivaProject
    set GLOG_logtostderr=1
    bazel-bin\mediapipe\examples\desktop\BivaProject\BivaProject.exe

    oder alternativ einfach die Datei "Ausführen.cmd" ausführen

## Quellen
    - eye detection: https://www.tutorialspoint.com/how-to-track-the-eye-in-opencv-using-cplusplus
    - bild laden anzeigen https://www.tutorialkart.com/opencv/python/opencv-python-read-png-images-with-transparency-channel/
    - Video (GIF) laden https://stackoverflow.com/questions/23177845/opencv-imread-doesnt-work/56252616#56252616
    - Kamera auswählen https://www.selfmadetechie.com/how-to-create-a-webcam-video-capture-using-opencv-c
    - Wie man Handzeichen ermittlen könnte https://gist.github.com/TheJLifeX/74958cc59db477a91837244ff598ef4a
    - Wie mediapipe in c++ funktioniert https://github.com/google/mediapipe/blob/master/mediapipe/examples/desktop/demo_run_graph_main.cc
    - Wie die Hände in MediaPipe angezeigt werden https://google.github.io/mediapipe/solutions/hands.html
    - Allgemeine git doku https://docs.opencv.org/4.x/
    - Handsup: https://stock.adobe.com/de/search?k=hands+up+surrender&asset_id=244328590 
    - GIGACHAD: https://production.listennotes.com/podcasts/the-gigachad-podcast-wCoeAVuYZS0-D1eQw4qkMcj.1400x1400.jpg 
    - ThumbUp: https://cdn.betterttv.net/emote/60c50c33f8b3f62601c3cdb1/3x.webp
    - Middle Finger: https://images.emojiterra.com/google/android-oreo/512px/1f595.png
    - Peace https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Peace_symbol.svg/2048px-Peace_symbol.svg.png
    - OKAY MEME: https://e7.pngegg.com/pngimages/171/972/png-clipart-saitama-cartoon-one-punch-man-saitama-ok-memes-saitama-ok.png
    - THE ROCK: https://media.tenor.com/IyweQyb3MhIAAAAi/the-rock-sus.gif
    - Little Devil: https://www.clipartmax.com/png/full/60-601728_purple-devil-emoji.png