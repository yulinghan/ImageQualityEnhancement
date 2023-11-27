cmake_minimum_required(VERSION 2.8)
PROJECT(akaze_demo)

FIND_PACKAGE(OpenCV REQUIRED)

IF(UNIX)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wno-unknown-pragmas -Wunused-variable -g -msse2 -msse3 -msse4 -ffast-math")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unknown-pragmas -Wunused-variable -g -msse2 -msse3 -msse4 -ffast-math")
ENDIF(UNIX)

SET(SRC
    main.cpp akaze/akaze_features.cpp akaze/akaze_fed.cpp akaze/akaze_nldiffusion_functions.cpp akaze/akaze_utils.cpp akaze/AKAZE.cpp
)

ADD_EXECUTABLE(${PROJECT_NAME} ${SRC})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS})
