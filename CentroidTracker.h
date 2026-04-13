#ifndef CENTROID_TRACKER_H
#define CENTROID_TRACKER_H

#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>

class CentroidTracker {
public:
    CentroidTracker(int maxDisappeared = 15, double maxDistance = 100.0);

    // main entry point - give bounding boxes, get tracked objects
    std::map<int, cv::Point2f> update(const std::vector<cv::Rect>& rects);

private:
    int nextObjectID;
    int maxDisappeared;  // frames before object is removed
    double maxDistance;   // max euclidean dist for matching

    std::map<int, cv::Point2f> objects;
    std::map<int, int> disappeared;

    void registerObject(cv::Point2f centroid);
    void deregisterObject(int objectID);
};

#endif
