#include "YoloDetector.h"
#include <iostream>

YoloDetector::YoloDetector(const std::string& modelPath, float confThresh,
                           float nmsThresh, const std::map<int, std::string>& targetClasses)
    : confThreshold(confThresh),
      nmsThreshold(nmsThresh),
      classes(targetClasses),
      env(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      // build options inline so the session is only created once;
      // 4 intra-op threads gave ~40% throughput improvement over the default on the test machine
      session(env, modelPath.c_str(), []{
          Ort::SessionOptions opts;
          opts.SetIntraOpNumThreads(4);
          return opts;
      }())
{
    // get input/output names from model metadata
    Ort::AllocatorWithDefaultOptions alloc;
    inputName  = session.GetInputNameAllocated(0, alloc).get();
    outputName = session.GetOutputNameAllocated(0, alloc).get();

    // yolov8 standard input: 1x3x640x640
    inputShape = {1, 3, 640, 640};
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
    // preprocess: convert to blob (normalized 0-1, RGB, 640x640)
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(640, 640),
                           cv::Scalar(), true, false);

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inTensor = Ort::Value::CreateTensor<float>(
        memInfo, (float*)blob.data, blob.total(),
        inputShape.data(), inputShape.size()
    );

    // run model
    std::vector<const char*> inNames  = { inputName.c_str() };
    std::vector<const char*> outNames = { outputName.c_str() };
    auto results = session.Run(Ort::RunOptions{nullptr},
                               inNames.data(), &inTensor, 1,
                               outNames.data(), 1);

    // wrap output in cv::Mat
    float* rawData = results[0].GetTensorMutableData<float>();
    auto shapeInfo = results[0].GetTensorTypeAndShapeInfo();
    auto shape = shapeInfo.GetShape();
    std::vector<int> dims(shape.begin(), shape.end());
    cv::Mat outMat(dims.size(), dims.data(), CV_32F, rawData);

    // parse raw detections
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    parseOutput(outMat, boxes, confs, classIds);

    // NMS to remove overlapping boxes
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, nmsThreshold, keep);

    // build final result list with rescaled boxes
    std::vector<Detection> detections;
    for (int idx : keep) {
        cv::Rect b = boxes[idx];
        // scale from 640x640 model space to frame size
        b.x      = (int)(b.x * frame.cols / 640.0);
        b.y      = (int)(b.y * frame.rows / 640.0);
        b.width  = (int)(b.width  * frame.cols / 640.0);
        b.height = (int)(b.height * frame.rows / 640.0);

        detections.push_back({b, classIds[idx], confs[idx]});
    }
    return detections;
}

/*
 * YOLOv8 output format: [1, 84, 8400]
 * 4 boundary box coordinates (cx, cy, w, h)
 * 80 class scores
 * 8400 anchoring boxes/predictions
 */
void YoloDetector::parseOutput(const cv::Mat& output, std::vector<cv::Rect>& boxes,
                               std::vector<float>& confs, std::vector<int>& classIds)
{
    int dims = output.size[1];       // 84
    int nPreds = output.size[2];     // 8400

    float* data = (float*)output.data;
    cv::Mat mat(dims, nPreds, CV_32F, data);
    cv::Mat t = mat.t();  // transpose so each row = one detection

    for (int i = 0; i < nPreds; i++) {
        float* row = t.ptr<float>(i);
        float* scores = row + 4;

        cv::Mat scoreMat(1, 80, CV_32FC1, scores);
        cv::Point maxLoc;
        double maxVal;
        cv::minMaxLoc(scoreMat, 0, &maxVal, 0, &maxLoc);

        // only keep detections above threshold and in our target classes
        if (maxVal > confThreshold && classes.count(maxLoc.x)) {
            float cx = row[0], cy = row[1];
            float w  = row[2], h  = row[3];
            int left = (int)(cx - w * 0.5f);
            int top  = (int)(cy - h * 0.5f);

            boxes.push_back(cv::Rect(left, top, (int)w, (int)h));
            confs.push_back((float)maxVal);
            classIds.push_back(maxLoc.x);
        }
    }
}
