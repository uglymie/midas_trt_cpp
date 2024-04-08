#ifndef MIDAS_H
#define MIDAS_H

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "util.h"
#include "transforms.h"

#include "opencv2/opencv.hpp"

using sample::gLogError;
using sample::gLogInfo;

class Midas
{
public:
    Midas();
    Midas(const std::string &trt_model_file,
          int net_size = 384);
    ~Midas();
    void loadEngine();
    std::unique_ptr<float> preprocessImage(cv::Mat image, int size = 512);
    bool runMonoDepth(std::unique_ptr<float> &input_data,
                      std::unique_ptr<float> &output_data);

    // void setImage(cv::Mat image);
    cv::Mat postProcessing(std::unique_ptr<float> &output_data);

private:
    std::string m_model_file;
    int m_net_size;
    int image_width;
    int image_height;

    nvinfer1::Dims m_input_dims;
    nvinfer1::Dims m_output_dims;
    util::UniquePtr<nvinfer1::ICudaEngine> m_engine;
};

#endif