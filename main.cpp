#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "json.hpp"
#include "CentroidTracker.h"
#include "YoloDetector.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

/*
 * Thread-safe queue implementation using mutex and condition_variable.
 * Designed for frame buffering between reader and tracker threads.
 */
template <typename T>
class SafeQueue {
    std::queue<T> q;
    std::mutex mtx;
    std::condition_variable cond;
public:
    void push(T item) {
        std::unique_lock<std::mutex> lk(mtx);
        q.push(std::move(item));
        cond.notify_one();
    }
    void pop(T& item) {
        std::unique_lock<std::mutex> lk(mtx);
        cond.wait(lk, [this]{ return !q.empty(); });
        item = std::move(q.front());
        q.pop();
    }
};

// --------------- data types ---------------

struct FrameData {
    cv::Mat frame;
    int number = 0;
    bool done = false;  // signals end of video
};

struct DetectionFrame {
    cv::Mat frame;
    int number = 0;
    bool done = false;
    std::vector<Detection> detections;
};

// --------------- config ---------------

struct Config {
    std::string modelPath    = "yolov8n.onnx";
    float confThreshold      = 0.3f;
    float nmsThreshold       = 0.4f;    // 0.5 left too many duplicates in testing
    int zoneYMin             = 380;
    int zoneYMax             = 420;
    int trackerMaxDisappeared  = 30;
    float trackerMaxDistance = 300.0f;
    float pixelsPerMeter    = 0.7f;    // rough estimate, depends on camera
    float maxSpeedKph       = 160.0f;
    float minSpeedKph       = 25.0f;
    float emaAlpha          = 0.1f;    // lower = smoother
    int minSpeedSamples     = 5;
    int frameWidth          = 800;
    int frameHeight         = 600;
    // coco80 ids: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
    std::map<int, std::string> vehicleClasses = {
        {2, "car"}, {3, "motorcycle"}, {5, "bus"}, {7, "truck"}
    };

    void loadFromFile(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "config not found: " << path << " (using defaults)" << std::endl;
            return;
        }
        try {
            json j;
            f >> j;
            if (j.contains("model")) {
                if (j["model"].contains("path")) modelPath = j["model"]["path"];
                if (j["model"].contains("confidence_threshold"))
                    confThreshold = j["model"]["confidence_threshold"];
                if (j["model"].contains("nms_threshold"))
                    nmsThreshold = j["model"]["nms_threshold"];
            }
            if (j.contains("zones")) {
                if (j["zones"].contains("y_min")) zoneYMin = j["zones"]["y_min"];
                if (j["zones"].contains("y_max")) zoneYMax = j["zones"]["y_max"];
            }
            if (j.contains("speed")) {
                auto& s = j["speed"];
                if (s.contains("pixels_per_meter")) pixelsPerMeter = s["pixels_per_meter"];
                if (s.contains("max_speed_kph"))    maxSpeedKph = s["max_speed_kph"];
                if (s.contains("min_speed_kph"))    minSpeedKph = s["min_speed_kph"];
                if (s.contains("ema_alpha"))         emaAlpha = s["ema_alpha"];
                if (s.contains("min_samples"))       minSpeedSamples = s["min_samples"];
            }
            if (j.contains("target_classes")) {
                vehicleClasses.clear();
                for (auto& [name, ids] : j["target_classes"].items()) {
                    for (int id : ids) vehicleClasses[id] = name;
                }
            }
            if (j.contains("tracker")) {
                if (j["tracker"].contains("max_disappeared")) trackerMaxDisappeared = j["tracker"]["max_disappeared"];
                if (j["tracker"].contains("max_distance")) trackerMaxDistance = j["tracker"]["max_distance"];
            }
        } catch (std::exception& e) {
            std::cerr << "config parse error: " << e.what() << std::endl;
        }
    }
};

// --------------- speed estimator ---------------

class SpeedEstimator {
    float pixPerMeter, alpha, minKph, maxKph;
    int minSamples;
    std::map<int, float> lastX;
    std::map<int, std::pair<float, int>> ema;  // {smoothed_speed, count}
public:
    SpeedEstimator(const Config& cfg)
        : pixPerMeter(cfg.pixelsPerMeter), alpha(cfg.emaAlpha),
          minKph(cfg.minSpeedKph), maxKph(cfg.maxSpeedKph),
          minSamples(cfg.minSpeedSamples) {}

    void update(int id, float cx, double fps) {
        if (!lastX.count(id)) {
            lastX[id] = cx;
            return;
        }
        float dx = std::abs(cx - lastX[id]);
        lastX[id] = cx;
        // filter movements under ~1m in physical space to reject bounding-box jitter;
        // threshold scales with pixPerMeter so it stays meaningful across camera setups
        if (dx < pixPerMeter) return;

        // convert pixels/frame -> km/h
        double dt = 1.0 / fps;
        float v = (float)((dx / pixPerMeter / dt) * 3.6);
        if (v < minKph || v > maxKph) return;  // out of reasonable range

        float prev = v;
        int n = 0;
        if (ema.count(id)) {
            prev = ema[id].first;
            n = ema[id].second;
        }
        // clamp sudden jumps after we have a baseline — ±15 km/h max change is empirical,
        // prevents single bad detections from corrupting the running estimate
        if (n > 3) {
            v = std::max(prev - 15.0f, std::min(prev + 15.0f, v));
        }
        float smooth = alpha * v + (1.0f - alpha) * prev;
        ema[id] = {smooth, n + 1};
    }

    // returns -1 if not enough samples yet
    float getSpeed(int id) const {
        if (!ema.count(id) || ema.at(id).second < minSamples) return -1.0f;
        return ema.at(id).first;
    }

    // get all estimated speeds
    const std::map<int, std::pair<float, int>>& getAllSpeeds() const { return ema; }
    int getMinSamples() const { return minSamples; }
};

// --------------- background builder ---------------

class BackgroundBuilder {
    std::vector<cv::Mat> samples;
    int interval;
    int counter = 0;
    static const int MAX_SAMPLES = 30;
public:
    BackgroundBuilder(int totalFrames)
        : interval(std::max(1, totalFrames / MAX_SAMPLES)) {}

    void addFrame(const cv::Mat& frame) {
        counter++;
        if (counter % interval == 0 && (int)samples.size() < MAX_SAMPLES) {
            samples.push_back(frame.clone());
        }
    }

    // Generates background using temporal mean estimation.
    // Optimization note: Per-pixel median would be more robust against outliers but mean estimation is prioritized here for acceptable performance without hardware acceleration
    bool generate(cv::Mat& output) {
        if (samples.empty()) return false;
        cv::Mat accum = cv::Mat::zeros(samples[0].size(), CV_32FC3);
        for (auto& s : samples) {
            cv::Mat tmp;
            s.convertTo(tmp, CV_32FC3);
            accum += tmp;
        }
        accum /= (double)samples.size();
        accum.convertTo(output, CV_8UC3);
        return true;
    }
};

// --------------- pipeline ---------------

void processVideo(const std::string& inputFile, const Config& cfg, bool showGui) {
    std::cout << "Processing: " << inputFile << std::endl;

    cv::VideoCapture cap(inputFile);
    if (!cap.isOpened()) {
        std::cerr << "failed to open: " << inputFile << std::endl;
        return;
    }

    if (showGui) {
        std::cout << "--- GUI MODE ---" << std::endl;
        std::cout << "runs yolo on every frame, will be slower than realtime" << std::endl;
        std::cout << "press ESC to stop" << std::endl;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    int totalFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    SafeQueue<FrameData> frameQueue;
    SafeQueue<DetectionFrame> detectQueue;
    cv::Size frameSize(cfg.frameWidth, cfg.frameHeight);

    // --- reader thread ---
    std::thread reader([&]() {
        int cnt = 0;
        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                frameQueue.push({cv::Mat(), cnt, true});
                break;
            }
            cnt++;
            cv::resize(frame, frame, frameSize);
            frameQueue.push({frame, cnt, false});
        }
    });

    // --- inference thread ---
    std::thread inference([&]() {
        YoloDetector detector(cfg.modelPath, cfg.confThreshold,
                              cfg.nmsThreshold, cfg.vehicleClasses);
        while (true) {
            FrameData fd;
            frameQueue.pop(fd);
            if (fd.done) {
                detectQueue.push({cv::Mat(), 0, true, {}});
                break;
            }
            auto dets = detector.detect(fd.frame);
            detectQueue.push({fd.frame, fd.number, false, dets});
        }
    });

    // --- main thread: tracking + counting ---
    CentroidTracker tracker(cfg.trackerMaxDisappeared, cfg.trackerMaxDistance);
    SpeedEstimator speedEst(cfg);
    BackgroundBuilder bgBuilder(totalFrames);

    int totalVehicles = 0;
    std::map<int, std::pair<std::string, std::string>> vehicleInfo;  // {dir, type}
    std::set<int> counted;
    std::map<int, float> prevCx;  // last known centroid X per track, for direction detection

    auto tStart = std::chrono::high_resolution_clock::now();
    int nFrames = 0;

    while (true) {
        DetectionFrame det;
        detectQueue.pop(det);
        if (det.done) break;

        nFrames++;
        bgBuilder.addFrame(det.frame);

        // convert Detection vector to Rect vector for tracker
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        for (auto& d : det.detections) {
            boxes.push_back(d.box);
            classIds.push_back(d.classId);
        }

        auto tracked = tracker.update(boxes);

        for (auto& [id, pt] : tracked) {
            float cx = pt.x, cy = pt.y;

            // find which detection box matches this tracked object
            std::string type = "car";
            cv::Rect box(cx - 20, cy - 20, 40, 40);  // fallback
            for (size_t i = 0; i < boxes.size(); i++) {
                if (boxes[i].contains(pt)) {
                    type = cfg.vehicleClasses.at(classIds[i]);
                    box = boxes[i];
                    break;
                }
            }

            speedEst.update(id, cx, fps);

            // count if crossing zone
            if (cy > cfg.zoneYMin && cy < cfg.zoneYMax && !counted.count(id)) {
                counted.insert(id);
                totalVehicles++;
                // derive direction from lateral movement; fall back to screen-side if no history
                std::string dir;
                if (prevCx.count(id) && std::abs(cx - prevCx.at(id)) > 2.0f) {
                    dir = (cx > prevCx.at(id)) ? "b" : "a";
                } else {
                    dir = (cx > cfg.frameWidth / 2) ? "b" : "a";
                }
                vehicleInfo[id] = {dir, type};
            }
            prevCx[id] = cx;

            // draw gui overlay
            if (showGui) {
                cv::rectangle(det.frame, box, cv::Scalar(0, 255, 0), 2);
                std::string label = type + " #" + std::to_string(id);
                float spd = speedEst.getSpeed(id);
                if (spd > 0) label += " " + std::to_string((int)spd) + "km/h";
                cv::putText(det.frame, label, cv::Point(box.x, box.y - 8),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }

        if (showGui) {
            cv::line(det.frame, cv::Point(0, cfg.zoneYMin),
                     cv::Point(det.frame.cols, cfg.zoneYMin), cv::Scalar(255, 0, 0), 1);
            cv::line(det.frame, cv::Point(0, cfg.zoneYMax),
                     cv::Point(det.frame.cols, cfg.zoneYMax), cv::Scalar(255, 0, 0), 1);
            cv::putText(det.frame, "Count: " + std::to_string(totalVehicles),
                        cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 0, 255), 2);
            cv::imshow("Traffic Analysis", det.frame);
            if (cv::waitKey(1) == 27) break;
        }
    }

    reader.join();
    inference.join();

    auto tEnd = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    // --------------- save output ---------------
    std::string outDir = fs::exists("/output") ? "/output" : "output";
    if (!fs::exists(outDir)) fs::create_directory(outDir);

    std::string baseName = fs::path(inputFile).stem().string();

    // background image
    cv::Mat bg;
    if (bgBuilder.generate(bg)) {
        std::string bgPath = outDir + "/" + baseName + "_background.jpg";
        cv::imwrite(bgPath, bg);
        std::cout << "background saved: " << bgPath << std::endl;
    }

    // tally per direction
    std::map<std::string, int> cntA = {{"car",0},{"truck",0},{"bus",0},{"motorcycle",0}};
    std::map<std::string, int> cntB = {{"car",0},{"truck",0},{"bus",0},{"motorcycle",0}};
    int nA = 0, nB = 0;
    for (auto& [vid, info] : vehicleInfo) {
        if (info.first == "a") { nA++; cntA[info.second]++; }
        else                   { nB++; cntB[info.second]++; }
    }

    // average speed per direction
    auto& speeds = speedEst.getAllSpeeds();
    float sumA = 0, sumB = 0;
    int spdNA = 0, spdNB = 0;
    for (auto& [vid, sp] : speeds) {
        if (sp.second >= speedEst.getMinSamples() && vehicleInfo.count(vid)) {
            if (vehicleInfo[vid].first == "a") { sumA += sp.first; spdNA++; }
            else                               { sumB += sp.first; spdNB++; }
        }
    }
    int avgA = spdNA > 0 ? (int)std::round(sumA / spdNA) : 0;
    int avgB = spdNB > 0 ? (int)std::round(sumB / spdNB) : 0;

    // write csv
    std::string csvPath = outDir + "/" + baseName + ".csv";
    std::ofstream csv(csvPath);
    csv << "total_vehicles,average_frame_time_ms,"
        << "direction_a_total,direction_a_cars,direction_a_trucks,direction_a_buses,direction_a_motorcycles,direction_a_avg_speed_kph,"
        << "direction_b_total,direction_b_cars,direction_b_trucks,direction_b_buses,direction_b_motorcycles,direction_b_avg_speed_kph" << std::endl;

    csv << totalVehicles << "," << elapsed / std::max(1, nFrames) << ","
        << nA << "," << cntA["car"] << "," << cntA["truck"] << "," << cntA["bus"] << "," << cntA["motorcycle"] << "," << avgA << ","
        << nB << "," << cntB["car"] << "," << cntB["truck"] << "," << cntB["bus"] << "," << cntB["motorcycle"] << "," << avgB << std::endl;

    std::cout << "results -> " << csvPath << std::endl;
    std::cout << "total: " << totalVehicles << " vehicles, "
              << elapsed / std::max(1, nFrames) << "ms/frame avg" << std::endl;
}

int main(int argc, char** argv) {
    bool gui = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gui" || arg == "-g") gui = true;
    }

    Config cfg;
    cfg.loadFromFile("config.json");

    std::string inDir = fs::exists("/input") ? "/input" : "input";
    bool found = false;
    for (auto& entry : fs::directory_iterator(inDir)) {
        std::string ext = entry.path().extension().string();
        if (ext == ".mp4" || ext == ".ts" || ext == ".avi") {
            processVideo(entry.path().string(), cfg, gui);
            found = true;
        }
    }
    if (!found) {
        std::cerr << "no video files in " << inDir << std::endl;
        return 1;
    }
    return 0;
}
