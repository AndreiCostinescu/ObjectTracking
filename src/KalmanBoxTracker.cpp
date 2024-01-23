#include "ObjectTracking/KalmanBoxTracker.h"

using namespace ObjectTracking;

int KalmanBoxTracker::count = 0;

KalmanBoxTracker::KalmanBoxTracker(cv::Mat const &bbox) {
    id = KalmanBoxTracker::count;
    KalmanBoxTracker::count++;

    kf = std::make_shared<cv::KalmanFilter>(KF_DIM_X, KF_DIM_Z);  // no control vector
    // state transition matrix (A), x(k) = A*x(k-1) + B*u(k) + w(k)
    kf->transitionMatrix = (cv::Mat_<float>(KF_DIM_X, KF_DIM_X) <<
                                1, 0, 0, 0, 1, 0, 0,
                                0, 1, 0, 0, 0, 1, 0,
                                0, 0, 1, 0, 0, 0, 1,
                                0, 0, 0, 1, 0, 0, 0,
                                0, 0, 0, 0, 1, 0, 0,
                                0, 0, 0, 0, 0, 1, 0,
                                0, 0, 0, 0, 0, 0, 1);
    // measurement matrix (H), z(k) = H*x(k) + v(k)
    kf->measurementMatrix = (cv::Mat_<float>(KF_DIM_Z, KF_DIM_X) <<
                                1, 0, 0, 0, 0, 0, 0,
                                0, 1, 0, 0, 0, 0, 0,
                                0, 0, 1, 0, 0, 0, 0,
                                0, 0, 0, 1, 0, 0, 0);
    // measurement noise covariance matrix (R), K(k) = P`(k)*Ct*inv(C*P`(k)*Ct + R)
    kf->measurementNoiseCov = (cv::Mat_<float>(KF_DIM_Z, KF_DIM_Z) <<
                                1,  0,  0,  0,
                                0,  1,  0,  0,
                                0,  0,  10, 0,
                                0,  0,  0,  10);
    // posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
    kf->errorCovPost = (cv::Mat_<float>(KF_DIM_X, KF_DIM_X) <<
                                10, 0,  0,  0,  0,   0,   0,
                                0,  10, 0,  0,  0,   0,   0,
                                0,  0,  10, 0,  0,   0,   0,
                                0,  0,  0,  10, 0,   0,   0,
                                0,  0,  0,  0,  1e4, 0,   0,
                                0,  0,  0,  0,  0,   1e4, 0,
                                0,  0,  0,  0,  0,   0,   1e4);
    // process noise covariance matrix (Q), P'(k) = A*P(k-1)*At + Q
    kf->processNoiseCov = (cv::Mat_<float>(KF_DIM_X, KF_DIM_X) <<
                                1, 0, 0, 0, 0,    0,    0,
                                0, 1, 0, 0, 0,    0,    0,
                                0, 0, 1, 0, 0,    0,    0,
                                0, 0, 0, 1, 0,    0,    0,
                                0, 0, 0, 0, 1e-2, 0,    0,
                                0, 0, 0, 0, 0,    1e-2, 0,
                                0, 0, 0, 0, 0,    0,    1e-4);
    // corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
    cv::vconcat(convertBBoxToZ(bbox),
                cv::Mat(KF_DIM_X - KF_DIM_Z, 1, CV_32F, cv::Scalar(0)),
                kf->statePost);
}

KalmanBoxTracker::~KalmanBoxTracker() = default;

cv::Mat KalmanBoxTracker::update(cv::Mat const &bbox) {
    timeSinceUpdate = 0;
    hitStreak += 1;
    xPost = kf->correct(convertBBoxToZ(bbox));
    cv::Mat bboxPost = convertXToBBox(xPost);
    return bboxPost;
}

cv::Mat KalmanBoxTracker::predict() {
    // bbox area (ds/dt + s) shouldn't be negative
    if (kf->statePost.at<float>(6, 0) + kf->statePost.at<float>(2, 0) <= 0)
        kf->statePost.at<float>(6, 0) *= 0;

    cv::Mat xPred = kf->predict();
    cv::Mat bboxPred = convertXToBBox(xPred);

    hitStreak = timeSinceUpdate > 0 ? 0 : hitStreak;
    timeSinceUpdate++;

    return bboxPred;
}

int KalmanBoxTracker::getFilterCount() {
    return KalmanBoxTracker::count;
}

int KalmanBoxTracker::getFilterId() const {
    return id;
}

int KalmanBoxTracker::getTimeSinceUpdate() const {
    return timeSinceUpdate;
}

int KalmanBoxTracker::getHitStreak() const {
    return hitStreak;
}

cv::Mat KalmanBoxTracker::getState() {
    return xPost.clone();
}

cv::Mat KalmanBoxTracker::convertBBoxToZ(cv::Mat const &bbox) {
    assert(bbox.rows == 1 && bbox.cols >= 4);
    float x = bbox.at<float>(0, 0);
    float y = bbox.at<float>(0, 1);
    float s = bbox.at<float>(0, 2) * bbox.at<float>(0, 3);
    float r = bbox.at<float>(0, 2) / bbox.at<float>(0, 3);

    return (cv::Mat_<float>(KF_DIM_Z, 1) << x, y, s, r);  // NOLINT(modernize-return-braced-init-list)
}

cv::Mat KalmanBoxTracker::convertXToBBox(cv::Mat const &state) {
    assert(state.rows == KF_DIM_X && state.cols == 1);
    float x = state.at<float>(0, 0);
    float y = state.at<float>(1, 0);
    auto w = float(sqrt(double(state.at<float>(2, 0) * state.at<float>(3, 0))));
    float h = state.at<float>(2, 0) / w;

    return (cv::Mat_<float>(1, 4) << x, y, w, h);  // NOLINT(modernize-return-braced-init-list)
}
