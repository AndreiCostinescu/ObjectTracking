/**
 * @desc:   kalmanfilter for boundary box tracking.
 *          opencv kalmanfilter documents:
 *              https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html
 * 
 * @author: lst
 * @date:   12/10/2021
 */

#pragma once

#include <cassert>
#include <cmath>
#include <memory>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

#define KF_DIM_X 7      // xc, yc, s, r, dxc/dt, dyc/dt, ds/dt
#define KF_DIM_Z 4      // xc, yc, s, r

namespace ObjectTracking {
    class KalmanBoxTracker {
        // variables
    public:
        using Ptr = std::shared_ptr<KalmanBoxTracker>;
    private:
        static int count;
        int id;
        int timeSinceUpdate = 0;
        int hitStreak = 0;
        std::shared_ptr<cv::KalmanFilter> kf = nullptr;
        cv::Mat xPost;

        // methods
    public:
        /**
         * @brief Kalman filter for bbox tracking
         * @param bbox bounding box, Mat(1, 4+) [xc, yc, w, h, ...]
         */
        explicit KalmanBoxTracker(cv::Mat const &bbox);

        virtual ~KalmanBoxTracker();

        KalmanBoxTracker(KalmanBoxTracker const &) = delete;

        void operator=(KalmanBoxTracker const &) = delete;

        /**
         * @brief updates the state vector with observed bbox. 
         * @param bbox  boundary box, Mat(1, 4+) [xc, yc, w, h, ...]
         * @return corrected bounding box estimate, Mat(1, 4)
         */
        cv::Mat update(cv::Mat const &bbox);

        /**
         * @brief advances the state vector and returns the predicted bounding box estimate. 
         * @return predicted bounding box, Mat(1, 4)
         */
        cv::Mat predict();

        static int getFilterCount();

        [[nodiscard]] int getFilterId() const;

        [[nodiscard]] int getTimeSinceUpdate() const;

        [[nodiscard]] int getHitStreak() const;

        cv::Mat getState();

    private:
        /**
         * @brief convert boundary box to measurement.
         * @param bbox boundary box (1, 4+) [x center, y center, width, height, ...]
         * @return measurement vector (4, 1) [x center; y center; scale/area; aspect ratio]
         */
        static cv::Mat convertBBoxToZ(cv::Mat const &bbox);

        /**
         * @brief convert state vector to boundary box.
         * @param state state vector (7, 1) (x center; y center; scale/area; aspect ratio; ...)
         * @return boundary box (1, 4) [x center, y center, width, height]
         */
        static cv::Mat convertXToBBox(cv::Mat const &state);
    };
}

