#include<iostream> // Input-output stream operations
#include<fstream> // File stream operations
#include<opencv2/opencv.hpp> // OpenCV library for computer vision tasks
#include<vector> // Standard template library for vectors
#include<dirent.h> // Directory handling
#include "csv_util.h" // Custom header file for CSV file handling
#include<cstring> // String manipulation operations

#include "pipeline.hpp"

using namespace std;

// Function to compute features for a specified region
vector<float> computeFeatures(cv::Mat& region_mask, cv::Mat& frame) {
    // Compute percent filled
    double filled_area = cv::countNonZero(region_mask);
    double total_area = region_mask.rows * region_mask.cols;
    double percent_filled = (filled_area / total_area) * 100.0;

    // Compute bounding box
    cv::Rect bbox = cv::boundingRect(region_mask);

    // Calculate bounding box ratio
    double bounding_box_ratio = static_cast<double>(bbox.width) / bbox.height;

    // Draw bounding box
    cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

    // Display percent filled on the frame
    stringstream ss;
    ss << "Filled: " << percent_filled << "%";
    cv::putText(frame, ss.str(), cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

    // Store features in a vector
    vector<float> featureVector;
    featureVector.push_back(percent_filled);
    featureVector.push_back(bounding_box_ratio);

    return featureVector;
}

int main(){
    cv::VideoCapture *capdev;
    
    capdev = new cv::VideoCapture(0,cv::CAP_AVFOUNDATION);
    if(!capdev -> isOpened()){
        cout<<"Unable to open video device"<<endl;

        return (-1);
    }

    // Get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                 (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT)
            );
    cout<<"Expected Size:"<<refS.width<<refS.height<<endl;

    // Identifies a window
    cv::namedWindow("Video", 1);
    cv::Mat frame;
    cv::Mat thresh,dst,seg;

    // Make the video frame gr

    for(;;){
        *capdev >> frame; // Get a new frame from the camera, rear as a stream

        if(frame.empty()){
            cout<<"Frame is empty"<<endl;
            break;
        }

        imshow("Video", frame);

        // See if there is a waiting keystroke
        char key = cv::waitKey(10);

        if(key == 'q'){
            break;
        }
        if(key == 't'){
            cv::Mat gray;
            // make the image grayscale
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            cv::Mat blurred;
            cv::GaussianBlur(gray, blurred, cv::Size(5,5),0);

            cv::Mat thresh;
            // threshold the image
            // threshold of 100
            // anything above threshold gets 255 **but** invert intensities
            // if pix < 100 then pix = 255:  white fg on black bg
            cv::threshold(blurred, thresh, 100, 255, cv::THRESH_BINARY_INV );

            imshow("Thresholded Image", thresh);

        }
        if(key=='d'){


            cv::cvtColor( frame, frame, cv::COLOR_BGR2GRAY);
            cv::threshold( frame, thresh, 100, 255, cv::THRESH_BINARY_INV );

            cv::morphologyEx(thresh, dst, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_RECT, cv::Point(3,3)));

            imshow("Dialation (Closing)", dst);

        }
        if(key=='e'){

            

            cv::cvtColor( frame, frame, cv::COLOR_BGR2GRAY);
            cv::threshold( frame, thresh, 100, 255, cv::THRESH_BINARY_INV );

            cv::morphologyEx(thresh, seg, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_RECT, cv::Point(3,3)));

            imshow("Erosion (Opening)", seg);

        }
        if(key=='s'){

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            cv::Mat blurred;
            cv::GaussianBlur(gray, blurred, cv::Size(5,5),0);

            cv::Mat thresh;
            cv::threshold(blurred, thresh, 100, 255, cv::THRESH_BINARY_INV );

            cv::Mat morph;
            cv::morphologyEx(thresh, morph, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

            cv::Mat labels, stats, centroids;
            int num_objects = cv::connectedComponentsWithStats(morph, labels, stats, centroids);

            cv::RNG rng(12345); // Random color generator


            for (int i = 1; i < num_objects; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > 100) { // Filter based on region size
                    cv::Mat region_mask = (labels == i);
                    cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                    frame.setTo(color, region_mask);
                }
            }

            imshow("Segmented Image", frame);

        }
        if (key == 'r') {
            vector<float> featureVector;

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, thresh, cv::Size(5, 5), 0);
            cv::threshold(thresh, thresh, 100, 255, cv::THRESH_BINARY_INV);
            cv::morphologyEx(thresh, seg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            cv::Mat labels, stats, centroids;
            int num_objects = cv::connectedComponentsWithStats(seg, labels, stats, centroids);
            for (int i = 1; i < num_objects; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > 100) { // Filter based on region size
                    cv::Mat region_mask = (labels == i);
                    vector<float> regionFeatures = computeFeatures(region_mask, frame);
                    featureVector.insert(featureVector.end(),regionFeatures.begin(), regionFeatures.end());
                }
            }

            imshow("Features", frame);
        }
        if(key=='n'){

            vector<float> featureVector;

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, thresh, cv::Size(5, 5), 0);
            cv::threshold(thresh, thresh, 100, 255, cv::THRESH_BINARY_INV);
            cv::morphologyEx(thresh, seg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            cv::Mat labels, stats, centroids;
            int num_objects = cv::connectedComponentsWithStats(seg, labels, stats, centroids);
            for (int i = 1; i < num_objects; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > 100) { // Filter based on region size
                    cv::Mat region_mask = (labels == i);
                    vector<float> regionFeatures = computeFeatures(region_mask, frame);
                    featureVector.insert(featureVector.end(),regionFeatures.begin(), regionFeatures.end());
                }
            }

            char csvfileName[256];
            strcpy(csvfileName, "featureVectors.csv");

            // Append features to CSV file
            char label[256];
            cout << "Enter the label name: ";
            cin >> label;
            append_image_data_csv(csvfileName, label, featureVector, 0);

            imshow("Features", frame);

        }
        if(key=='c'){
            char featureVectorFile[256];
            strcpy(featureVectorFile, "featureVectors.csv");

             // Reading feature vectors from file
            vector<char *> filenames; // Vector to store filenames
            vector<vector<float>> data; // Vector to store feature vectors
            if(read_image_data_csv(featureVectorFile, filenames, data)!=0){
                cout<<"Error: Unable to read feature vector file"<<endl;
                exit(-1);
            }


            // Compute feature vector for the current stream
            vector<float> featureVector;

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, thresh, cv::Size(5, 5), 0);
            cv::threshold(thresh, thresh, 100, 255, cv::THRESH_BINARY_INV);
            cv::morphologyEx(thresh, seg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
            cv::Mat labels, stats, centroids;
            int num_objects = cv::connectedComponentsWithStats(seg, labels, stats, centroids);
            for (int i = 1; i < num_objects; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) > 100) { // Filter based on region size
                    cv::Mat region_mask = (labels == i);
                    vector<float> regionFeatures = computeFeatures(region_mask, frame);
                    featureVector.insert(featureVector.end(),regionFeatures.begin(), regionFeatures.end());
                }
            }


            // Calculating distance and sorting
            vector<pair<string, float>> distances; // Vector to store filename-distance pairs
            for(size_t i=0; i<filenames.size(); ++i){
                float distance = SSD(featureVector, data[i]); // Computing distance
                distances.emplace_back(filenames[i], distance); // Storing filename-distance pair
            }

            // Sorting filenames based on distance
            sort(distances.begin(), distances.end(), [](const pair<string, float>& a, const pair<string, float>& b) {
                return a.second < b.second;
            });

            
            
            cout<<"Top matched Label:"<<endl;
            cout<<distances[0].first<<" (Distance: "<< distances[0].second << ")"<<endl;

            }

        }

    
    delete capdev;

    return 0;
}