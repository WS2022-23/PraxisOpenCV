cmake_minimum_required(VERSION 3.0.0)
project(BiVaPraktikum VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(OPENCV_PATH ${OpenCV_DIR} )
set(SRC_PATH ${CMAKE_SOURCE_DIR})
configure_file(config.hpp.in config.hpp)

add_executable(BiVaPraktikum main.cpp)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries( BiVaPraktikum ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
