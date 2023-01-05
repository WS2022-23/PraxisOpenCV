cd mediapipe
set BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
set BAZEL_VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C:/development/python.exe" mediapipe/examples/desktop/BivaProject
set GLOG_logtostderr=1
bazel-bin\mediapipe\examples\desktop\BivaProject\BivaProject.exe