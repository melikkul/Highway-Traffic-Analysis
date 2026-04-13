#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    cv::Rect box;
    int classId;
    float confidence;
};

/*
 * Wraps ONNXRuntime inference for YOLOv8.
 * Handles session init, preprocessing, inference, and postprocessing.
 */
class YoloDetector {
public:
    YoloDetector(const std::string& modelPath, float confThresh,
                 float nmsThresh, const std::map<int, std::string>& targetClasses);
    ~YoloDetector() = default;

    // run detection on frame, returns filtered + nms'd results
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    float confThreshold;
    float nmsThreshold;
    std::map<int, std::string> classes;

    // onnx stuff
    Ort::Env env;
    Ort::Session session;
    std::string inputName;
    std::string outputName;
    std::vector<int64_t> inputShape;

    // parse raw yolov8 output tensor
    void parseOutput(const cv::Mat& output, std::vector<cv::Rect>& boxes,
                     std::vector<float>& confs, std::vector<int>& classIds);
};

#endif
