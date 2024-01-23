#include <cassert>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include "sort.h"

#include <AndreiUtils/utils.hpp>
#include <PerceptionData/containers/PersonDetectionContainer.h>
#include <VisualPerception/utils/Perception.h>
#include <VisualPerception/utils/utils.h>
#include <VisualPerception/inputs/data/ColorData.h>

namespace fs = std::filesystem;

using std::cout;
using std::endl;
using std::ifstream;
using std::stoi;
using std::stof;
using std::vector;
using std::pair;
using std::string;
using std::map;
using std::tuple;

using cv::Mat;
using cv::Mat_;
using cv::Rect;
using cv::Scalar;
using cv::RNG;
using cv::Point;

using sort::Sort;

using namespace PerceptionData;
using namespace VisualPerception;

auto constexpr MAX_COLORS = 2022;
vector<Scalar> COLORS;

vector<string> split(const string &s, char delim) {
    std::istringstream iss(s);
    vector<string> ret;
    string item;
    while (getline(iss, item, delim))
        ret.push_back(item);

    return ret;
}

// (seq info, [(image, detection), ...])
tuple<map<string, string>, vector<pair<Mat, Mat>>> getInputData(string dataFolder, bool useGT = false) {
    if (*dataFolder.end() != '/') dataFolder += '/';

    ifstream ifs;
    ifs.open(dataFolder + "seqinfo.ini");
    assert(ifs.is_open());
    map<string, string> mp;
    string s;
    while (getline(ifs, s)) {
        size_t pos = s.find('=');
        if (pos != string::npos) {
            string key = s.substr(0, pos);
            string val = s.substr(pos + 1, s.size() - pos);
            mp[key] = val;
        }
    }
    ifs.close();
    assert(mp.find("imDir") != mp.end());
    assert(mp.find("frameRate") != mp.end());
    assert(mp.find("seqLength") != mp.end());

    // get file list
    vector<string> imgPaths;
    for (const auto &entry: fs::directory_iterator(dataFolder + mp["imDir"])) {
        imgPaths.push_back(entry.path());
    }
    std::sort(imgPaths.begin(), imgPaths.end());
    assert(imgPaths.size() == std::stoi(mp["seqLength"]));

    vector<pair<Mat, Mat>> pairs(imgPaths.size(), {Mat(0, 0, CV_32F), Mat(0, 6, CV_32F)});

    // read images
    for (int i = 0; i < imgPaths.size(); ++i) {
        pairs[i].first = cv::imread(imgPaths[i]);
    }

    // read detections
    if (useGT) ifs.open(dataFolder + "/gt/gt.txt");
    else ifs.open(dataFolder + "/det/det.txt");
    assert (ifs.is_open());
    float x0, y0, w, h, score;
    int frameId, objId;
    while (getline(ifs, s)) {
        vector<string> ss = split(s, ',');
        frameId = stoi(ss[0]);
        objId = stoi(ss[1]);
        x0 = stof(ss[2]);
        y0 = stof(ss[3]);
        w = stof(ss[4]);
        h = stof(ss[5]);
        score = stof(ss[6]);
        Mat bbox = (Mat_<float>(1, 6) << x0 + w / 2, y0 + h / 2, w, h, score, objId);
        cv::vconcat(pairs[frameId - 1].second, bbox, pairs[frameId - 1].second);
    }

    return std::make_tuple(mp, pairs);
}

void draw(Mat &img, Mat const &bboxes) {
    float xc, yc, w, h, score, dx, dy;
    int trackerId;
    string sScore;
    for (int i = 0; i < bboxes.rows; ++i) {
        xc = bboxes.at<float>(i, 0);
        yc = bboxes.at<float>(i, 1);
        w = bboxes.at<float>(i, 2);
        h = bboxes.at<float>(i, 3);
        dx = bboxes.at<float>(i, 6);
        dy = bboxes.at<float>(i, 7);
        trackerId = int(bboxes.at<float>(i, 8));

        cv::rectangle(img, Rect(int(xc - w / 2), int(yc - h / 2), int(w), int(h)), COLORS[trackerId % MAX_COLORS], 2);
        cv::putText(img, std::to_string(trackerId), Point(int(xc - w / 2), int(yc - h / 2 - 4)),
                    cv::FONT_HERSHEY_PLAIN, 1.5, COLORS[trackerId % MAX_COLORS], 2);
        cv::arrowedLine(img, Point(int(xc), int(yc)), Point(int(xc + 5 * dx), int(yc + 5 * dy)),
                        COLORS[trackerId % MAX_COLORS], 4);
    }
}

void oldDemo(int argc, char **argv) {
    cout << "SORT demo" << endl;
    if (argc != 2) {
        cout << "usage: ./demo_sort [data folder], e.g. ./demo_sort ../data/TUD-Campus/" << endl;
        return;
    }
    string dataFolder = argv[1];
    // string dataFolder = "../data/TUD-Stadtmitte/";

    // read image and detections
    cout << "Read image and detections..." << endl;
    auto [seqInfo, motPairs] = getInputData(dataFolder);
    float fps = std::stof(seqInfo["frameRate"]);

    // tracking
    cout << "Tracking..." << endl;
    Sort::Ptr mot = std::make_shared<Sort>(1, 3, 0.3f);
    cv::namedWindow("SORT", cv::WindowFlags::WINDOW_NORMAL);
    for (auto [image, boundingBoxesDetections]: motPairs) {
        Mat trackedBoundingBoxes = mot->update(boundingBoxesDetections);

        // show result
        draw(image, trackedBoundingBoxes);
        cv::imshow("SORT", image);
        cv::waitKey(int(3000.0 / fps));
    }

    cout << "Done" << endl;
}

void demo() {
    Perception p("Perception");
    cout << "Before initialize" << endl;
    p.initialize();

    cout << "Before perceptionInitialization" << endl;
    if (!p.perceptionInitialization()) {
        return;
    }

    // <{outputImage: outputImage, inputImageClone: inputImageClone, originalDepth: originalDepth}, depthIntrinsics, {openpose: skeletons, darknet: yoloDetections}>
    VisualPerceptionOutputData output;

    int exit = 0;
    cv::namedWindow("Color - Input", cv::WINDOW_NORMAL);
    cv::resizeWindow("Color - Input", 1352, 1013);
    cv::namedWindow("Color - Output", cv::WINDOW_NORMAL);
    cv::resizeWindow("Color - Output", 1352, 1013);
    //*/
    cv::Mat const *colorData;
    cv::Mat *outputColorData;
    cout << std::setprecision(15);
    Sort::Ptr tracker = std::make_shared<Sort>(1, 3, 0.3f);
    for (; exit == 0;) {
        // cout << "In while at " << count << endl;
        if (!p.perceptionIteration()) {
            cout << "Perception Iteration returned false" << endl;
            exit = 1;
        }

        if (output.getInputDataIfContains<ColorData>(colorData)) {
            imshow("Color - Input", *colorData);
            // cout << "Color timestamp: " << convertChronoToStringWithSubseconds(output.getInput<ColorData>()->getTimestamp()) << endl;
        } else {
            cv::waitKey(1);
            continue;
        }
        auto const &imageSize = output.getInput<ColorData>()->getIntrinsics().size;

        cv::Mat detectionBoundingBoxes;  // N x 6; N = nr detected people ; 6 = [center_x, center_y, w, h, score, class]
        p.getOutput(output);
        for (auto const &device: output.getDevicesList()) {
            PerceptionDataContainer *deviceOutputContainer;
            if (output.getDeviceDataIfContains<PersonDetectionContainer>(deviceOutputContainer, device)) {
                auto persons = *(PersonDetectionContainer *) deviceOutputContainer;
                for (auto const &personData: persons) {
                    auto const &skeletonKeypoints = personData.second.getSkeleton().getJointImagePositions();
                    float minX = (float) imageSize.w, maxX = 0, minY = (float) imageSize.h, maxY = 0;
                    // collect bounding box data from person
                    for (auto const &keyPoint: skeletonKeypoints) {
                        auto const &x = keyPoint.second.x();
                        auto const &y = keyPoint.second.y();
                        if (AndreiUtils::less(minX, x)) {
                            minX = x;
                        }
                        if (AndreiUtils::greater(maxX, x)) {
                            maxX = x;
                        }
                        if (AndreiUtils::less(minY, y)) {
                            minY = y;
                        }
                        if (AndreiUtils::greater(maxY, y)) {
                            maxY = y;
                        }
                    }
                    float w = maxX - minX, h = maxY - minY;
                    Mat bbox = (Mat_<float>(1, 6) << minX + w / 2, minY + h / 2, w, h, personData.second.getConfidence(), 0);
                    cv::vconcat(detectionBoundingBoxes, bbox, detectionBoundingBoxes);
                }
            }
        }
        Mat trackedBoundingBoxes = tracker->update(detectionBoundingBoxes);

        if (output.getOutputDataIfContains<ColorData>(outputColorData)) {
            imshow("Color - Output", *outputColorData);
        }

        // show result
        draw(*outputColorData, trackedBoundingBoxes);
        cv::imshow("SORT RESULT", *outputColorData);

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            cout << "Manual stop" << endl;
            exit = 1;
        }
    }

    cv::destroyAllWindows();
    p.finish();
}

int main(int argc, char **argv) {
    // visual perception initialization
    setConfigurationParametersDirectory("../config/");

    initializeOpenposeForVisualPerception();
    initializeRealsenseForVisualPerception();

    // generate colors
    RNG rng(MAX_COLORS);
    for (size_t i = 0; i < MAX_COLORS; ++i) {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        COLORS.push_back(color);
    }

    // oldDemo(argc, argv);
    demo();

    return 0;
}