#include "transforms.h"

//******************Resize*******************//
Resize::Resize(
    int width,
    int height,
    bool resize_target,
    bool keep_aspect_ratio,
    int ensure_multiple_of,
    std::string resize_method,
    int image_interpolation_method) : __width(width),
                                                       __height(height),
                                                       __resize_target(resize_target),
                                                       __keep_aspect_ratio(keep_aspect_ratio),
                                                       __multiple_of(ensure_multiple_of),
                                                       __resize_method(resize_method),
                                                       __image_interpolation_method(image_interpolation_method)
{
}

Resize::~Resize()
{
}

int Resize::constrain_to_multiple_of(int x, int min_val, int max_val)
{
    int y = (cvRound(x / __multiple_of) * __multiple_of);

    if (max_val != -1 && y > max_val)
    {
        y = (cvFloor(x / __multiple_of) * __multiple_of);
    }

    if (y < min_val)
    {
        y = (cvCeil(x / __multiple_of) * __multiple_of);
    }

    return y;
}

std::pair<int, int> Resize::get_size(int input_width, int input_height)
{
    double scale_height = static_cast<double>(__height) / input_height;
    double scale_width = static_cast<double>(__width) / input_width;

    if (__keep_aspect_ratio)
    {
        if (__resize_method == "lower_bound")
        {
            if (scale_width > scale_height)
            {
                scale_height = scale_width;
            }
            else
            {
                scale_width = scale_height;
            }
        }
        else if (__resize_method == "upper_bound")
        {
            if (scale_width < scale_height)
            {
                scale_height = scale_width;
            }
            else
            {
                scale_width = scale_height;
            }
        }
        else if (__resize_method == "minimal")
        {
            if (std::abs(1 - scale_width) < std::abs(1 - scale_height))
            {
                scale_height = scale_width;
            }
            else
            {
                scale_width = scale_height;
            }
        }
        else
        {
            throw std::invalid_argument("resize_method " + __resize_method + " not implemented");
        }
    }

    int new_height, new_width;

    if (__resize_method == "lower_bound")
    {
        new_height = constrain_to_multiple_of(scale_height * input_height, 0, __height);
        new_width = constrain_to_multiple_of(scale_width * input_width, 0, __width);
    }
    else if (__resize_method == "upper_bound")
    {
        new_height = constrain_to_multiple_of(scale_height * input_height, 0, __height);
        new_width = constrain_to_multiple_of(scale_width * input_width, 0, __width);
    }
    else if (__resize_method == "minimal")
    {
        new_height = constrain_to_multiple_of(scale_height * input_height);
        new_width = constrain_to_multiple_of(scale_width * input_width);
    }
    else
    {
        throw std::invalid_argument("resize_method " + __resize_method + " not implemented");
    }

    return {new_width, new_height};
}

void Resize::operator()(cv::Mat image, cv::Mat &resized_image)
{
    // Get new size
    auto size = get_size(image.cols, image.rows);

    // Resize image
    cv::resize(image, resized_image, cv::Size(size.first, size.second), 0, 0, __image_interpolation_method);

    // Additional processing for resize_target, PrepareForNet(), etc.
    if(__resize_target)
    {

    }
}

//******************NormalizeImage*******************//
NormalizeImage::NormalizeImage(/* args */)
{
}

NormalizeImage::~NormalizeImage()
{
}

//******************PrepareForNet*******************//
PrepareForNet::PrepareForNet(/* args */)
{
}

PrepareForNet::~PrepareForNet()
{
}

void PrepareForNet::operator()(/* input data or sample */)
{
    // Retrieve image from sample
    cv::Mat image  /* extract image from sample, assuming it's a cv::Mat */;

    // Transpose the image (channels, rows, cols) -> (rows, cols, channels)
    cv::Mat transposed_image;
    cv::transpose(image, transposed_image);

    // Convert image data to float32 and make it contiguous
    cv::Mat float_image;
    transposed_image.convertTo(float_image, CV_32F);
    cv::Mat contiguous_image = float_image.clone();

    // Update the sample with the processed image
    // sample["image"] = contiguous_image;

    // Additional processing for "mask", "disparity", "depth", etc.
    // ...
}
