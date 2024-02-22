#include<iostream> // Standard input/output stream
#include<vector>    // Standard vector container
#include<opencv2/opencv.hpp> // OpenCV library
#include<algorithm> // Standard algorithms library

#include "csv_util.h"

using namespace std;


float SSD(const vector<float>& feature1, const vector<float>& feature2){

    float ssd = 0, dx;

    for(size_t i=0;i<feature1.size();++i){
        dx = feature1[i] - feature2[i];
        ssd += dx*dx;
    }

    return ssd;
}

// // Function to compute features for a specified region
// void computeFeatures(cv::Mat& region_mask, cv::Mat& frame) {
//     // Compute percent filled
//     double filled_area = cv::countNonZero(region_mask);
//     double total_area = region_mask.rows * region_mask.cols;
//     double percent_filled = (filled_area / total_area) * 100.0;

//     // Compute bounding box
//     cv::Rect bbox = cv::boundingRect(region_mask);

//     // Draw bounding box
//     cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

//     // Display percent filled
//     stringstream ss;
//     ss << "Filled: " << percent_filled << "%";
//     cv::putText(frame, ss.str(), cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
// }

// vector<float> computeFeatures(cv::Mat& region_mask, cv::Mat& frame) {
//     // Compute percent filled
//     double filled_area = cv::countNonZero(region_mask);
//     double total_area = region_mask.rows * region_mask.cols;
//     double percent_filled = (filled_area / total_area) * 100.0;

//     // Compute bounding box
//     cv::Rect bbox = cv::boundingRect(region_mask);

//     // Calculate bounding box ratio
//     double bounding_box_ratio = static_cast<double>(bbox.width) / bbox.height;

//     // Draw bounding box
//     cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

//     // Store features in a vector
//     std::vector<float> featureVector;
//     featureVector.push_back(percent_filled);
//     featureVector.push_back(bounding_box_ratio);


//     return featureVector;
// }
