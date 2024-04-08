#ifndef TRANSFORMS_H
#define TRANSFORMS_H
#include <iostream>
#include <opencv2/opencv.hpp>

/// @brief Resize 类是一个图像调整的工具类，用于将输入的图像、掩码（mask）、目标（target）等调整到指定的大小。
class Resize
{

public:
    Resize(
        int width,
        int height,
        bool resize_target = true,
        bool keep_aspect_ratio = false,
        int ensure_multiple_of = 1,
        std::string resize_method = "lower_bound",
        int image_interpolation_method = cv::INTER_AREA);
    ~Resize();
int Resize::constrain_to_multiple_of(int x, int min_val = 0, int max_val = -1);
std::pair<int, int> Resize::get_size(int input_width, int input_height);
void Resize::operator()(cv::Mat image, cv::Mat &resized_image);

private:
    int __width;    // 期望的输出宽度
    int __height;   // 期望的输出高度
    bool __resize_target;   // 指定是否调整完整样本（图像、掩码、目标），默认为 True
    bool __keep_aspect_ratio;   // 指定是否保持输入样本的纵横比。如果为 True，则输出样本可能不会严格符合给定的宽度和高度，具体调整行为取决于参数 resize_method。默认为 False。
    int __multiple_of;      // 确保输出宽度和高度是指定参数的倍数，默认为 1
    std::string __resize_method;    // 指定调整图像大小的方法
    int __image_interpolation_method;   // 图像插值方法

};


class NormalizeImage
{
private:
    /* data */
public:
    NormalizeImage(/* args */);
    ~NormalizeImage();
};

class PrepareForNet
{
private:
    /* data */
public:
    PrepareForNet(/* args */);
    ~PrepareForNet();
    void operator()(/* input data or sample */);
};










#endif // TRANSFORMS_H