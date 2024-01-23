#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include <ObjectTracking/ObjectTracker.h>

#include <AndreiUtils/utils.hpp>
#include <AndreiUtils/utilsFiles.h>
#include <PerceptionData/containers/PersonDetectionContainer.h>
#include <VisualPerception/utils/Perception.h>
#include <VisualPerception/utils/utils.h>
#include <VisualPerception/inputs/data/ColorData.h>

using namespace cv;
using namespace PerceptionData;
using namespace std;
using namespace VisualPerception;

using ObjectTracking::ObjectTracker;

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
    for (const auto &entry: AndreiUtils::listDirectoryFiles(dataFolder + mp["imDir"])) {
        imgPaths.push_back(entry);
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
    ObjectTracker::Ptr mot = std::make_shared<ObjectTracker>(1, 3, 0.3f);
    cv::namedWindow("SORT", cv::WindowFlags::WINDOW_NORMAL);
    for (auto [image, boundingBoxesDetections]: motPairs) {
        Mat trackedBoundingBoxes = mot->update(boundingBoxesDetections);

        // show result
        ObjectTracker::draw(image, trackedBoundingBoxes);
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
    ObjectTracker::Ptr tracker = std::make_shared<ObjectTracker>(1, 3, 0.3f);
    for (; exit == 0;) {
        // cout << "In while at " << count << endl;
        if (!p.perceptionIteration()) {
            cout << "Perception Iteration returned false" << endl;
            exit = 1;
        }
        p.getOutput(output);

        if (output.getInputDataIfContains<ColorData>(colorData)) {
            imshow("Color - Input", *colorData);
            // cout << "Color timestamp: " << convertChronoToStringWithSubseconds(output.getInput<ColorData>()->getTimestamp()) << endl;
        } else {
            cv::waitKey(1);
            cout << "No color data!" << endl;
            continue;
        }
        auto const &imageSize = output.getInput<ColorData>()->getIntrinsics().size;

        // N x 6; N = nr detected people ; 6 = [center_x, center_y, w, h, score, class]
        cv::Mat detectionBoundingBoxes(0, 6, CV_32F);
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
                        if (AndreiUtils::less(x, minX)) {
                            minX = x;
                        }
                        if (AndreiUtils::greater(x, maxX)) {
                            maxX = x;
                        }
                        if (AndreiUtils::less(y, minY)) {
                            minY = y;
                        }
                        if (AndreiUtils::greater(y, maxY)) {
                            maxY = y;
                        }
                    }
                    float w = maxX - minX, h = maxY - minY;
                    Mat bbox = (Mat_<float>(1, 6) << minX + w / 2, minY +
                                                                   h / 2, w, h, personData.second.getConfidence(), 0);
                    cv::vconcat(detectionBoundingBoxes, bbox, detectionBoundingBoxes);
                }
            }
        }
        Mat trackedBoundingBoxes = tracker->update(detectionBoundingBoxes);

        if (output.getOutputDataIfContains<ColorData>(outputColorData)) {
            imshow("Color - Output", *outputColorData);
        }

        // show result
        ObjectTracker::draw(*outputColorData, trackedBoundingBoxes, true);
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

    // oldDemo(argc, argv);
    demo();

    return 0;
}