/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>      // for std::accumulate
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    constexpr int dataBufferSize = 2;                   // no. of images which are held in memory (ring buffer) at the same time
    RingBuffer<DataFrame, dataBufferSize> dataBuffer;   // list of data frames which are held in memory at the same time
    bool bVis = false;                                  // visualize results

    // For the last tasks regarding performance, we need some variables to keep track
    // of the data over all images.
    vector<cv::KeyPoint> all_keypoints;

    /* MAIN LOOP OVER ALL IMAGES */
    size_t imgIndex = 0;
    for (; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = "BRISK";
        if (argc > 1) {
            detectorType = argv[1];
            cout << "Setting the keypoint detector type based on the command line: " << detectorType << "\n";
        }

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0) {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0) {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST") == 0) {
            detKeypointsFast(keypoints, imgGray, false);
        }
        else if (detectorType.compare("BRISK") == 0) {
            detKeypointsBrisk(keypoints, imgGray, false);
        }
        else if (detectorType.compare("ORB") == 0) {
            detKeypointsOrb(keypoints, imgGray, false);
        }
        else if (detectorType.compare("AKAZE") == 0) {
            detKeypointsAkaze(keypoints, imgGray, false);
        }
        else if (detectorType.compare("SIFT") == 0) {
            detKeypointsSift(keypoints, imgGray, false);
        } else {
            std::cerr << "\n *** Error: You requested an invalid keypoint detector by providing " << detectorType;
            std::cerr << "\n *** Allowed keypoint detectors are: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT\n\n";
            return -1;
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        vector<cv::KeyPoint> keypoints_on_vehicle;
        if (bFocusOnVehicle)
        {
            std::copy_if(keypoints.begin(), keypoints.end(), std::back_inserter(keypoints_on_vehicle), [&vehicleRect](const cv::KeyPoint& kpt) {
                return vehicleRect.contains(cv::Point2i((int) kpt.pt.x, (int) kpt.pt.y));
            });
            std::swap(keypoints_on_vehicle, keypoints);
        }

        // For the performance evaluation, draw the keypoints and save an image.
        // The image files are used for the writeup / readme.
        if (true) {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = detectorType + " Results";
            cv::namedWindow(windowName, 6);
            imshow(windowName, visImage);
            cv::imwrite(dataPath + "writeup/keypoints/" + detectorType + ".jpg", visImage);
            // cv::waitKey(0);
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorType = "BRISK"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        DescriptorType dt;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, dt);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
            string descriptorType; // DES_BINARY, DES_HOG
            switch(dt) {
                case DescriptorType::BINARY:
                    descriptorType = "DES_BINARY";
                    break;
                case DescriptorType::HOG:
                    descriptorType = "DES_HOG";
                    break;
            }
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                // cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

        // Collect some data for the performance statistics.
        for (const cv::KeyPoint &kpt : keypoints) {
            all_keypoints.push_back(kpt);
        }

    } // eof loop over all images

    cout << "*** Performance statistics summary ***\n";
    cout << "There have been " << imgIndex << " images.\n";
    float avg_num_keypoints_per_image = (float) all_keypoints.size() / ((float) imgIndex);
    cout << "Average number of keypoints (on the vehicle) per image: " << avg_num_keypoints_per_image << "\n";
    // Compute the mean
    float avg_size = std::accumulate(all_keypoints.begin(), all_keypoints.end(), 0.0f, [](const float &val, const cv::KeyPoint &kpt) {
            return (val + kpt.size);
        }) / ((float) all_keypoints.size());
    // Compute the sample standard deviation. See for instance here:
    // https://www.statisticshowto.com/probability-and-statistics/descriptive-statistics/sample-variance/
    float std_dev_size = std::accumulate(all_keypoints.begin(), all_keypoints.end(), 0.0f, [&avg_size](const float &val, const cv::KeyPoint &kpt) {
            return (val + (kpt.size - avg_size) * (kpt.size - avg_size));
        }) / ((float) (all_keypoints.size() - 1));
    std_dev_size = std::sqrt(std_dev_size);
    cout << "Average size of the keypoints (on the vehicle): " << avg_size << "\n";
    cout << "Size of the keypoints (on the vehicle) standard deviation: " << std_dev_size << "\n";

    return 0;
}
