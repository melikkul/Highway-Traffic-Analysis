FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/app/onnxruntime/lib

# build tools + opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopencv-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# onnxruntime
RUN wget -q -O onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz \
    && tar -xzf onnxruntime.tgz \
    && rm onnxruntime.tgz \
    && mv onnxruntime-linux-x64-1.20.1 onnxruntime

# copy source
COPY CMakeLists.txt \
     main.cpp \
     CentroidTracker.h \
     CentroidTracker.cpp \
     YoloDetector.h \
     YoloDetector.cpp \
     json.hpp \
     yolov8n.onnx \
     config.json \
     ./

# build
RUN mkdir build && cd build && cmake .. && make -j$(nproc)

ENTRYPOINT ["./build/main"]
