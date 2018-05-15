cmake_minimum_required(VERSION 2.8.11)

if(APPLE)
#    set(CMAKE_C_COMPILER /usr/local/opt/llvm/bin/clang)
#    set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)
#    set(CMAKE_C_COMPILER gcc-7)
#    set(CMAKE_CXX_COMPILER g++-7)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
    #GCC detects CPUID
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")

    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-missing-field-initializers")
    set(EXTRA_CXX_FLAGS -Weffc++ -Wswitch-default -Wfloat-equal -Wconversion -Wimplicit-fallthrough)
else()
    set(CMAKE_C_COMPILER gcc)
    set(CMAKE_CXX_COMPILER g++)
endif()

project(TextDetector)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CXX_FLAGS "-fPIC ${CFLAGS_WARN} -std=c++1y ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 14)

set(OPENCV_CPP_REQUIRED_DEPS opencv_core opencv_flann opencv_imgproc
    opencv_highgui opencv_ml opencv_video opencv_objdetect opencv_photo
    opencv_calib3d opencv_stitching opencv_videostab opencv_xfeatures2d opencv_features2d
)

ocv_check_dependencies(${OPENCV_CPP_REQUIRED_DEPS})

set(BLOBS_LIB_SOURCES
    BlobsLib/blob.cpp
    BlobsLib/BlobContour.cpp
    BlobsLib/BlobOperators.cpp
    BlobsLib/BlobResult.cpp
    BlobsLib/ComponentLabeling.cpp
)

set(BLOBS_LIB_HEADERS
    BlobsLib/blob.h
    BlobsLib/BlobContour.h
    BlobsLib/BlobLibraryConfiguration.h
    BlobsLib/BlobOperators.h
    BlobsLib/BlobResult.h
    BlobsLib/ComponentLabeling.h
)

set(TEXT_DETECTOR_SOURCES
    TextDetector.cpp
    ImageBinarization.cpp
)

set(TEXT_DETECTOR_HEADERS
    TextDetector.h
    ImageBinarization.cpp
    Singleton.h
)

add_library(TextDetector SHARED ${TEXT_DETECTOR_HEADERS} ${TEXT_DETECTOR_SOURCES} ${BLOBS_LIB_SOURCES} ${BLOBS_LIB_HEADERS})
target_link_libraries(TextDetector pthread ${OpenCV_LIBS})

add_executable(main_save_boxes_to_file main_save_boxes_to_file.cpp)
target_link_libraries(main_save_boxes_to_file TextDetector)
