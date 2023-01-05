# BildVerarbeitung_OpenCV
In diesem Repository befindet sich das Praktikumsprojekt aus der Veranstaltung Bildverarbeitung 1 im WiSe 2022/23


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

    Bei WinSDK vorher schauen ob es schon welche gibt und diese dann löschen (so hab ich des zumindestens gemacht)
    

## Bauen und Ausführen des Programmes

    Bevor es mit dem Bauen losgehen kann muss das Projekt erstmal mit Cmake gebaut werden
    Nachdem das Bauen erfolgreich war muss die config.h Datei in den Ordner "mediapipe/examples/desktop/BivaProject" 
        ersetzt werden da hier Pfade dabei sind

    set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
    set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C:/development/python.exe" mediapipe/examples/desktop/BivaProject
    set GLOG_logtostderr=1
    bazel-bin\mediapipe\examples\desktop\BivaProject\BivaProject.exe

## Quellen
    - eye detection: https://www.tutorialspoint.com/how-to-track-the-eye-in-opencv-using-cplusplus
    - bild laden anzeigen https://www.tutorialkart.com/opencv/python/opencv-python-read-png-images-with-transparency-channel/
    - Wie man Handzeichen ermittlen könnte https://gist.github.com/TheJLifeX/74958cc59db477a91837244ff598ef4a
    - Wie mediapipe in c++ funktioniert https://github.com/TheJLifeX/mediapipe


![QR-Code https://www.youtube.com/watch?v=hGlyFc79BUE](https://cdn.discordapp.com/attachments/706821180426027058/960587179514560652/unknown1.png)