#include "midas.h"

Midas::Midas(const std::string &trt_model_file, int net_size)
    : m_model_file(trt_model_file), m_net_size(net_size), m_engine(nullptr)
{
    loadEngine();
}

Midas::Midas()
{
}

Midas::~Midas()
{
}

void Midas::loadEngine()
{
    std::ifstream engineFile(m_model_file, std::ios::binary); // 打开二进制文件流
    if (engineFile.fail())
    {
        return;
    }

    engineFile.seekg(0, std::ifstream::end); // 将文件指针移到文件的末尾，以获取文件的大小
    auto fsize = engineFile.tellg();         // 获取文件的大小
    engineFile.seekg(0, std::ifstream::beg); // 将文件指针重新设置到文件的开头，以便后续可以读取文件的内容

    std::vector<char> engineData(fsize);       // 存储整个文件的内容
    engineFile.read(engineData.data(), fsize); // 从文件中读取二进制数据并将其存储到 engineData 向量中

    util::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())}; // 创建了TensorRT的运行时对象
    m_engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));                         // 反序列化TensorRT引擎
    assert(m_engine.get() != nullptr);                                                                         // 确保引擎成功反序列化并且不为空
}

std::unique_ptr<float> Midas::preprocessImage(cv::Mat image, int size)
{
    image_width = image.cols;
    image_height = image.rows;
    // BGR to RGB
    cv::Mat rgb_image, resize_image;
    if (image.channels() == 1)
        cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    // // 调整图像大小，同时保持纵横比(等比例缩放)
    // float aspect_ratio = static_cast<float>(image.cols) / image.rows;
    // int resize_width, resize_height;
    // if (aspect_ratio >= 1.0)
    // {
    //     resize_width = size;
    //     resize_height = static_cast<int>(size / aspect_ratio);
    // }
    // else
    // {
    //     resize_height = size;
    //     resize_width = static_cast<int>(size * aspect_ratio);
    // }
    Resize img_resize(size, size, true, false, 32, "upper_bound", cv::INTER_CUBIC);
    img_resize(rgb_image, resize_image);
    
    // cv::imshow("resize_image", resize_image);
    // cv::waitKey(0);
    const int channel = 3;
    std::unique_ptr<float> image_data = std::unique_ptr<float>{new float[m_net_size * m_net_size * 3]};
    for (int c = 0; c < channel; c++)
    {
        for (int h = 0; h < size; h++)
        {
            for (int w = 0; w < size; w++)
            {
                image_data.get()[c * size * size + h * size + w] = resize_image.at<cv::Vec3b>(h, w)[c] / 255.0;
            }
        }
    }

    return image_data;
}

bool Midas::runMonoDepth(std::unique_ptr<float> &input_data,
                         std::unique_ptr<float> &output_data)
{
    // 创建TensorRT的执行上下文对象
    auto context = util::UniquePtr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    auto input_idx = m_engine->getBindingIndex("onnx::Sub_0"); // 获取输入绑定的索引
    if (input_idx == -1)
    {
        return false;
    }
    // 使用断言来确保输入绑定的数据类型是 kFLOAT 即浮点数
    assert(m_engine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
    // 定义输入数据为4维，有1个样本，3个通道（RGB图像），以及指定的高度和宽度
    auto input_dims = nvinfer1::Dims4{1, 3, m_net_size, m_net_size};
    context->setBindingDimensions(input_idx, input_dims);             // 将输入绑定的维度设置为 output_data_dims，确保输入数据的维度与模型的输入绑定匹配
    auto input_size = util::getMemorySize(input_dims, sizeof(float)); // 基于维度和数据类型来计算输入数据内存大小
    std::cout << "input_size: " << input_size << std::endl;     // input_size: 442,368 * 4

    auto output_idx = m_engine->getBindingIndex("2991"); // 获取输出绑定的索引  5318
    if (output_idx == -1)
    {
        return false;
    }
    assert(m_engine->getBindingDataType(output_idx) == nvinfer1::DataType::kFLOAT);
    auto output_dims = context->getBindingDimensions(output_idx);       // 获取输出绑定的维度(前提是模型中维度已知)
    auto output_size = util::getMemorySize(output_dims, sizeof(float)); // 计算输出数据内存大小
    std::cout << "output_size: " << output_size << std::endl;     // output_size: 147,456 * 4

    // 在CUDA设备上分配内存，以用于输入和输出数据的绑定
    void *input_mem{nullptr};                              // 存储输入数据的CUDA内存指针
    if (cudaMalloc(&input_mem, input_size) != cudaSuccess) // 分配输入数据的CUDA内存
    {
        gLogError << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }
    void *output_mem{nullptr};                               // 存储输出数据的CUDA内存指针
    if (cudaMalloc(&output_mem, output_size) != cudaSuccess) // 分配输出数据的CUDA内存
    {
        gLogError << "ERROR: output cuda memory allocation failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }

    // 从文件中读取图像数据，并进行归一化处理
    // const std::vector<float> mean{0.485f, 0.456f, 0.406f};      // 均值
    // const std::vector<float> stddev{0.229f, 0.224f, 0.225f};    // 标准差
    // auto input_image{util::RGBImageReader(input_filename, output_data_dims, mean, stddev)};
    // input_image.read();
    // auto input_buffer = input_image.process();

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) // 创建CUDA流, 用于异步执行CUDA操作的对象
    {
        gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

    // 复制图像数据到输入绑定内存中
    if (cudaMemcpyAsync(input_mem, input_data.get(), input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of input failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }

    // 执行 TensorRT 推理 将输入数据传递给模型，然后从模型中获取输出数据
    void *bindings[] = {input_mem, output_mem};                  // 创建一个指向设备内存的指针数组
    bool status = context->enqueueV2(bindings, stream, nullptr); // 执行推断
    if (!status)
    {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    output_data = std::unique_ptr<float>{new float[output_size]}; // 输出主机内存数据
    // 将模型的输出数据从设备内存复制到主机内存(异步)
    if (cudaMemcpyAsync(output_data.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of output failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream); // 同步等待CUDA流中的所有异步操作完成

    // Free CUDA resources
    cudaFree(input_mem);
    cudaFree(output_mem);
    return true;
}

cv::Mat Midas::postProcessing(std::unique_ptr<float> &output_data)
{
    cv::Mat1f depth_img(m_net_size, m_net_size);

    for (int h = 0; h < m_net_size; h++)
    {
        for (int w = 0; w < m_net_size; w++)
        {
           depth_img.at<float>(h, w) = output_data.get()[h * m_net_size + w];
        }
    }

    // for (int w= 0; w < m_net_size; w++)
    // {
    //     for (int h = 0; h < m_net_size; h++)
    //     {
    //        depth_img.at<float>(w, h) = output_data.get()[h * m_net_size + w];
    //     }
    // }
    cv::resize(depth_img, depth_img, cv::Size(image_width, image_height), cv::INTER_CUBIC);

    int h = 719, w = 1279;
    float depth_value = depth_img.at<float>(h, w);
    // 打印像素值
    std::cout << "Depth at (" << h << ", " << w << "): " << depth_value << std::endl;

    bool grayscale = true;
    int bits = 2;
    // 如果不使用灰度，强制设置 bits 为 1
    if (!grayscale) {
        bits = 1;
    }

    // 处理非有限的深度值
    // cv::Mat validDepth = depth_img.clone();
    // cv::Mat zeroMask = cv::Mat::zeros(depth_img.size(), depth_img.type());
    // cv::compare(depth_img, zeroMask, validDepth, cv::CMP_EQ);

    // 计算深度范围
    double depthMin, depthMax;
    cv::minMaxLoc(depth_img, &depthMin, &depthMax);

    // 计算深度图的缩放因子，确保深度范围在合适的位数内
    int maxVal = (1 << (8 * bits)) - 1;

    cv::Mat output;
    if (depthMax - depthMin > std::numeric_limits<double>::epsilon()) {
        output = maxVal * (depth_img - depthMin) / (depthMax - depthMin);
    } else {
        output = cv::Mat::zeros(depth_img.size(), depth_img.type());
    }

    // 如果不是灰度，应用 Inverno 色图
    if (!grayscale) {
        cv::applyColorMap(output, output, cv::COLORMAP_INFERNO);
    }
    if (bits == 1) {
        output.convertTo(output, CV_8U);
    } else if (bits == 2) {
        output.convertTo(output, CV_16U);
    }
    
    return output;
}
