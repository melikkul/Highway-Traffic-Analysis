#include "CentroidTracker.h"
#include <numeric>
#include <limits>

CentroidTracker::CentroidTracker(int maxDisappeared, double maxDistance)
    : nextObjectID(0), maxDisappeared(maxDisappeared), maxDistance(maxDistance) {}

void CentroidTracker::registerObject(cv::Point2f centroid) {
    objects[nextObjectID] = centroid;
    disappeared[nextObjectID] = 0;
    nextObjectID++;
}

void CentroidTracker::deregisterObject(int objectID) {
    objects.erase(objectID);
    disappeared.erase(objectID);
}

/*
* Matches existing tracks to new detections using greedy nearest-neighbor matching 
* based on Euclidean distance. Sufficiently performant for sparse/medium density highway traffic.
*/
std::map<int, cv::Point2f> CentroidTracker::update(const std::vector<cv::Rect>& rects) {
    // no detections ---> increment disappear counter for everything
    if (rects.empty()) {
        auto it = disappeared.begin();
        while (it != disappeared.end()) {
            it->second++;
            if (it->second > maxDisappeared) {
                objects.erase(it->first);
                it = disappeared.erase(it);
            } else {
                ++it;
            }
        }
        return objects;
    }

    // compute centroids of input rectangles
    std::vector<cv::Point2f> inputCentroids(rects.size());
    for (size_t i = 0; i < rects.size(); i++) {
        float cx = rects[i].x + rects[i].width / 2.0f;
        float cy = rects[i].y + rects[i].height / 2.0f;
        inputCentroids[i] = cv::Point2f(cx, cy);
    }

    if (objects.empty()) {
        // nothing tracked yet, just register all
        for (auto& c : inputCentroids) registerObject(c);
    } else {
        // collect existing ids and positions
        std::vector<int> ids;
        std::vector<cv::Point2f> pts;
        for (auto& p : objects) {
            ids.push_back(p.first);
            pts.push_back(p.second);
        }

        // compute pairwise distance matrix
        size_t nOld = pts.size();
        size_t nNew = inputCentroids.size();
        std::vector<std::vector<double>> dist(nOld, std::vector<double>(nNew));
        for (size_t i = 0; i < nOld; i++) {
            for (size_t j = 0; j < nNew; j++) {
                dist[i][j] = std::hypot(
                    pts[i].x - inputCentroids[j].x,
                    pts[i].y - inputCentroids[j].y
                );
            }
        }

        // greedy matching: process rows sorted by their min distance
        std::vector<int> rowOrder(nOld);
        std::iota(rowOrder.begin(), rowOrder.end(), 0);
        std::sort(rowOrder.begin(), rowOrder.end(), [&](int a, int b) {
            return *std::min_element(dist[a].begin(), dist[a].end())
                 < *std::min_element(dist[b].begin(), dist[b].end());
        });

        std::set<int> usedRows, usedCols;

        for (int r : rowOrder) {
            // find closest unmatched column
            int bestCol = -1;
            double bestDist = std::numeric_limits<double>::max();
            for (size_t j = 0; j < nNew; j++) {
                if (!usedCols.count(j) && dist[r][j] < bestDist) {
                    bestDist = dist[r][j];
                    bestCol = j;
                }
            }
            if (bestCol == -1) continue; // nothing available
            if (bestDist > maxDistance) continue; // too far

            objects[ids[r]] = inputCentroids[bestCol];
            disappeared[ids[r]] = 0;
            usedRows.insert(r);
            usedCols.insert(bestCol);
        }

        // old objects that didn't match anything
        for (size_t i = 0; i < nOld; i++) {
            if (!usedRows.count(i)) {
                disappeared[ids[i]]++;
                if (disappeared[ids[i]] > maxDisappeared) {
                    deregisterObject(ids[i]);
                }
            }
        }

        // new detections that didn't match any old object
        for (size_t j = 0; j < nNew; j++) {
            if (!usedCols.count(j)) {
                registerObject(inputCentroids[j]);
            }
        }
    }

    return objects;
}
