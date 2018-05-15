g++ \
    -L/usr/local/lib \
    -pthread \
    -std=c++11 \
    ../BlobsLib/*.cpp debug.cpp \
    -lopencv_core \
    -lopencv_highgui \
    -lopencv_imgcodecs \
    -lopencv_imgproc \
    -o debug.out && ./debug.out