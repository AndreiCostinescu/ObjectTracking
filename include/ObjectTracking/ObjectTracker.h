/**
 * @desc:   C++ implementation of SORT.
 *          Bewley Alex "Simple, online, and realtime tracking of multiple objects in a video sequence", 
 *          http://arxiv.org/abs/1602.00763, 2016.
 *          
 * @author: lst
 * @date:   12/10/2021
 */

#pragma once

#include <memory>
#include <ObjectTracking/KuhnMunkres.h>
#include <ObjectTracking/KalmanBoxTracker.h>

namespace ObjectTracking {
    using std::shared_ptr;
    using std::vector;
    using std::pair;
    using std::tuple;
    using std::make_tuple;
    using std::make_shared;
    using kuhn_munkres::KuhnMunkres;
    using kuhn_munkres::Vec2f;
    using kuhn_munkres::Vec1f;

    using TypeMatchedPairs = vector<pair<int, int>>;   // first: detected id, second: predicted id
    using TypeLostDets = vector<int>;
    using TypeLostPreds = vector<int>;
    using TypeAssociate = tuple<TypeMatchedPairs, TypeLostDets, TypeLostPreds>;

    class ObjectTracker {
        // variables
    public:
        using Ptr = std::shared_ptr<ObjectTracker>;
    private:
        int maxAge;         // tracker's maximal unmatch count
        int minHits;        // tracker's minimal match count
        float iouThresh;    // IoU threshold
        vector<KalmanBoxTracker::Ptr> trackers;
        KuhnMunkres::Ptr km = nullptr;
        static int const maxColors;
        static vector<cv::Scalar> colors;
        static bool colorsInitialized;

        // methods
    public:
        explicit ObjectTracker(int maxAge = 1, int minHits = 3, float iouThresh = 0.3);

        virtual ~ObjectTracker();

        ObjectTracker(const ObjectTracker &) = delete;

        ObjectTracker &operator=(const ObjectTracker &) = delete;

        /**
         * @brief bbox tracking in SORT, this method must be called once for each frame even with empty detections, 
         *        the number of objects retured may differ from the number of detections provided.
         * @param bboxesDet detections, Mat(M, 6) with the format [[xc,yc,w,h,score,class_id];[...];...]
         * @return matched bboxes, Mat(N, 9) with the format [[xc,yc,w,h,score,class_id,dx,dy,tracker_id];[...];...].
         */
        cv::Mat update(cv::Mat const &bboxesDet);

        static void draw(cv::Mat &img, cv::Mat const &bboxes, bool withScore = false);

    private:
        /** 
         * @brief check if NAN value in Mat
         * @param mat input Matrix 
         * @return any NAN value in Matrix or not.
         */
        template<typename Tp>
        static bool isAnyNan(cv::Mat const &mat) {
            for (auto it = mat.begin<Tp>(); it != mat.end<Tp>(); ++it)
                if (*it != *it) {
                    return true;
                }
            return false;
        }

        /**
         * @brief data associate in SORT
         * @param bboxesDet detected bboxes, Mat(M, 4+)
         * @param bboxesPred predicted bboxes, Mat(N, 4+)
         * @return associate tuple (matched pairs, lost detections, lost predictions)
         */
        TypeAssociate dataAssociate(cv::Mat const &bboxesDet, cv::Mat const &bboxesPred);

        /**
         * @brief IoU of bboxes
         * @param bboxesA input bboxes A, Mat(M, 4+)
         * @param bboxesB another input bboxes B, Mat(N, 4+)
         * @return M x N matrix, value(i, j) means IoU of A(i) and B(j)
         */
        static cv::Mat getIouMatrix(cv::Mat const &bboxesA, cv::Mat const &bboxesB);

        static void initializeColors();
    };
}

