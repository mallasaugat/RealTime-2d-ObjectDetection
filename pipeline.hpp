#ifndef PIPELINE_H
#define PIPELINE_H

float SSD(const std::vector<float>& feature1, const std::vector<float>& feature2);

std::vector<float> computeFeatures(cv::Mat& region_mask, cv::Mat& frame);

#endif