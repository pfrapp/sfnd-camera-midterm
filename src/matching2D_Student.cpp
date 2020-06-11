#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {

        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    // Set up the detector parameters for the Harris detector.
    // The values have been taken from the previous exercise 'cornerness_harris'.

    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    double t = (double)cv::getTickCount();

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Now we locate salient points (keypoints).
    // The vector is already given as input argument.

    // Maximum permissible overlap between features in percent.
    // Used during non-maximum suppression.
    // This is an intersection-over-union value.
    // 0.0 means that we do not tolerate any overlap at all.
    double maxOverlap = 0.0;

    int keypoint_candidate_counter = 0;

    // Loop over every pixel in the Harris response matrix.
    // Use dst_norm (not dst_norm_scaled)
    for (int row = 0; row < dst_norm.rows; row++) {
        for (int col = 0; col < dst_norm.cols; col++) {
            
            // Harris response.
            // The response in 'dst_norm' assumes values between 0 and 255 (inclusive).
            int response = (int) dst_norm.at<float>(row, col);

            // Check if the response value is over the threshold of 'minResponse'.
            if (response > minResponse) {

                // Create a keypoint candidate and set some of its parameters.
                cv::KeyPoint keyPtCandidate;
                // Note: For points, x goes to the right, like an x-axis.
                // Therefore x corresponds to the column.
                // y corresponds to the row.
                keyPtCandidate.pt = cv::Point2f(col, row);
                // The size of the keypoint corresponds to the size of the Sobel operator
                // that was used in the Harris algorithm.
                keyPtCandidate.size = 2 * apertureSize;
                keyPtCandidate.response = response;


                // Now perform the acutal NMS (non-maximum suppression).
                bool bOverlap = false;

                // Loop over all existing keypoints and look for overlaps
                // with the new candidate keypoint.
                for (cv::KeyPoint& confirmedKeyPt : keypoints) {
                    // Compute the intersection over union, which is 0 for no overlap,
                    // and which is 1 for complete overlap.
                    double keypoint_overlap_iou = cv::KeyPoint::overlap(keyPtCandidate, confirmedKeyPt);

                    // If this keypoint candidate has an overlap with an existing keypoint
                    if (keypoint_overlap_iou > maxOverlap) {

                        // Keypoint candidate overlaps with existing keypoint.
                        bOverlap = true;

                        // If the candidate keypoint has a higher response intensity,
                        // replace the existing confirmed keypoint with the candidate.
                        if (keyPtCandidate.response > confirmedKeyPt.response) {
                            confirmedKeyPt = keyPtCandidate;
                            break;
                        }

                    }

                } // end of loop over all existing keypoints

                // If there was not significant overlap (as defined by maxOverlap), add
                // this keypoint candidate to the list of confirmed keypoints.
                if (!bOverlap) {
                    // Add the keypoint candidate to the list of keypoints.
                    keypoints.push_back(keyPtCandidate);
                }


            } // end of threshold check 'response > minResponse'

        } // end of loop over columns
    } // end of loop over rows

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis) {
    // Draw all keypoints
        string windowName = "Harris Corner Detection Results (Keypoints)";
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage,
            cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {

}

void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {

}

void detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {

}

void detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {

}

void detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {

}

