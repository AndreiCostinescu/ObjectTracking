#include "ObjectTracking/ObjectTracker.h"
#include <iostream>

using namespace ObjectTracking;

int const ObjectTracker::maxColors = 2022;
std::vector<cv::Scalar> ObjectTracker::colors;
bool ObjectTracker::colorsInitialized = false;

ObjectTracker::ObjectTracker(int maxAge, int minHits, float iouThresh)
        : maxAge(maxAge), minHits(minHits), iouThresh(iouThresh) {
    km = std::make_shared<KuhnMunkres>();
    if (!ObjectTracker::colorsInitialized) {
        ObjectTracker::initializeColors();
    }
}

ObjectTracker::~ObjectTracker() = default;

cv::Mat ObjectTracker::update(cv::Mat const &bboxesDet) {
    assert(bboxesDet.rows >= 0 && bboxesDet.cols == 6); // detections, [xc, yc, w, h, score, class_id]

    // predictions used in data association, [xc, yc, w, h, score, class_id]
    cv::Mat bboxesPred(0, 6, CV_32F, cv::Scalar(0));
    // bounding boxes estimate, [xc, yc, w, h, score, class_id, vx, vy, tracker_id]
    cv::Mat bboxesPost(0, 9, CV_32F, cv::Scalar(0));

    // kalman bbox tracker predict
    for (auto it = trackers.begin(); it != trackers.end();) {
        cv::Mat bboxPred = (*it)->predict();   // Mat(1, 4)
        if (isAnyNan<float>(bboxPred))
            trackers.erase(it);     // remove the NAN value and corresponding tracker
        else {
            cv::hconcat(bboxPred, cv::Mat(1, 2, CV_32F, cv::Scalar(0)), bboxPred);   // Mat(1, 6)
            cv::vconcat(bboxesPred, bboxPred, bboxesPred);  // Mat(N, 6)
            ++it;
        }
    }

    TypeAssociate asTuple = dataAssociate(bboxesDet, bboxesPred);
    TypeMatchedPairs matchedDetPred = std::get<0>(asTuple);
    TypeLostDets lostDets = std::get<1>(asTuple);
    TypeLostPreds lostPreds = std::get<2>(asTuple);

    // update matched trackers with assigned detections
    for (auto pair: matchedDetPred) {
        int detInd = pair.first;
        int predInd = pair.second;
        cv::Mat bboxPost = trackers[predInd]->update(bboxesDet.rowRange(detInd, detInd + 1));

        if (trackers[predInd]->getHitStreak() >= minHits) {
            float score = bboxesDet.at<float>(detInd, 4);
            int classId = (int) bboxesDet.at<float>(detInd, 5);
            float dx = trackers[predInd]->getState().at<float>(4, 0);
            float dy = trackers[predInd]->getState().at<float>(5, 0);
            int trackerId = trackers[predInd]->getFilterId();
            cv::Mat tailData = (cv::Mat_<float>(1, 5) << score, classId, dx, dy, trackerId);
            cv::hconcat(bboxPost, tailData, bboxPost);
            cv::vconcat(bboxesPost, bboxPost, bboxesPost);  // Mat(N, 9)
        }
    }

    // remove dead trackers
    trackers.erase(std::remove_if(trackers.begin(), trackers.end(),
                                  [&](KalmanBoxTracker::Ptr const &kbt) -> bool {
                                      return kbt->getTimeSinceUpdate() > this->maxAge;
                                  }), trackers.end());

    // create and initialize new trackers for unmatched detections
    for (int lostInd: lostDets) {
        cv::Mat lostBbox = bboxesDet.rowRange(lostInd, lostInd + 1);
        trackers.push_back(make_shared<KalmanBoxTracker>(lostBbox));
    }

    return bboxesPost;
}

void ObjectTracker::draw(cv::Mat &img, cv::Mat const &bboxes, bool withScore) {
    float xc, yc, w, h, score, dx, dy;
    int trackerId;
    std::string sScore;
    for (int i = 0; i < bboxes.rows; ++i) {
        xc = bboxes.at<float>(i, 0);
        yc = bboxes.at<float>(i, 1);
        w = bboxes.at<float>(i, 2);
        h = bboxes.at<float>(i, 3);
        score = bboxes.at<float>(i, 4);
        dx = bboxes.at<float>(i, 6);
        dy = bboxes.at<float>(i, 7);
        trackerId = int(bboxes.at<float>(i, 8));

        cv::rectangle(img, cv::Rect(int(xc - w / 2), int(yc - h / 2), int(w), int(h)),
                      ObjectTracker::colors[trackerId % ObjectTracker::maxColors], 2);
        sScore = std::to_string(trackerId);
        if (withScore) {
            sScore += ": " + std::to_string(score);
        }
        cv::putText(img, sScore, cv::Point(int(xc - w / 2), int(yc - h / 2 - 4)),
                    cv::FONT_HERSHEY_PLAIN, 1.5, ObjectTracker::colors[trackerId % ObjectTracker::maxColors], 2);
        cv::arrowedLine(img, cv::Point(int(xc), int(yc)), cv::Point(int(xc + 5 * dx), int(yc + 5 * dy)),
                        ObjectTracker::colors[trackerId % ObjectTracker::maxColors], 4);
    }
}

TypeAssociate ObjectTracker::dataAssociate(cv::Mat const &bboxesDet, cv::Mat const &bboxesPred) {
    TypeMatchedPairs matchedDetPred;
    TypeLostDets lostDets;
    TypeLostPreds lostPreds;

    // initialize
    for (int i = 0; i < bboxesDet.rows; ++i) {
        lostDets.push_back(i);  // size M
    }
    for (int j = 0; j < bboxesPred.rows; ++j) {
        lostPreds.push_back(j); // size N
    }

    // nothing detected or predicted
    if (bboxesDet.rows == 0 || bboxesPred.rows == 0) {
        return make_tuple(matchedDetPred, lostDets, lostPreds);
    }

    // compute IoU matrix
    cv::Mat iouMat = getIouMatrix(bboxesDet, bboxesPred);   // Mat(M, N)

    // Kuhn Munkres assignment algorithm
    Vec2f costMatrix(iouMat.rows, Vec1f(iouMat.cols, 0.0f));
    for (int i = 0; i < iouMat.rows; ++i) {
        for (int j = 0; j < iouMat.cols; ++j) {
            costMatrix[i][j] = 1.0f - iouMat.at<float>(i, j);
        }
    }
    auto indices = km->compute(costMatrix);

    // find matched pairs and lost detect and predict
    for (auto [detInd, predInd]: indices) {
        matchedDetPred.emplace_back(detInd, predInd);
        lostDets.erase(remove(lostDets.begin(), lostDets.end(), detInd), lostDets.end());
        lostPreds.erase(remove(lostPreds.begin(), lostPreds.end(), predInd), lostPreds.end());
    }

    return make_tuple(matchedDetPred, lostDets, lostPreds);
}

cv::Mat ObjectTracker::getIouMatrix(cv::Mat const &bboxesA, cv::Mat const &bboxesB) {
    assert(bboxesA.cols >= 4 && bboxesB.cols >= 4);
    int numA = bboxesA.rows;
    int numB = bboxesB.rows;
    cv::Mat iouMat(numA, numB, CV_32F, cv::Scalar(0.0));

    cv::Rect re1, re2;
    for (int i = 0; i < numA; ++i) {
        for (int j = 0; j < numB; ++j) {
            re1.x = int(bboxesA.at<float>(i, 0) - bboxesA.at<float>(i, 2) / 2.0);
            re1.y = int(bboxesA.at<float>(i, 1) - bboxesA.at<float>(i, 3) / 2.0);
            re1.width = int(bboxesA.at<float>(i, 2));
            re1.height = int(bboxesA.at<float>(i, 3));
            re2.x = int(bboxesB.at<float>(j, 0) - bboxesB.at<float>(j, 2) / 2.0);
            re2.y = int(bboxesB.at<float>(j, 1) - bboxesB.at<float>(j, 3) / 2.0);
            re2.width = int(bboxesB.at<float>(j, 2));
            re2.height = int(bboxesB.at<float>(j, 3));

            iouMat.at<float>(i, j) = float((re1 & re2).area()) / (float((re1 | re2).area()) + FLT_EPSILON);
        }
    }

    return iouMat;
}

void ObjectTracker::initializeColors() {
    // generate colors
    cv::RNG rng(ObjectTracker::maxColors);
    for (size_t i = 0; i < ObjectTracker::maxColors; ++i) {
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        ObjectTracker::colors.emplace_back(color);
    }
    ObjectTracker::colorsInitialized = true;
}