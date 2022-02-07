#include "onnxruntime_c_api.h"
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

/*
https://github.com/leimao/ONNX-Runtime-Inference/blob/main/src/inference.cpp
*/

template<typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

std::vector<std::string> readLabels(std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

template<typename T>
void parse_output(const Ort::Value& out_data, unsigned long result_size, const std::vector<std::string>& labels)
{
    auto result_ptr = out_data.GetTensorData<T>();
    auto best_prediction = std::max_element(result_ptr, result_ptr + result_size);
    std::cout << "\n\nprediction_score: " << *best_prediction << std::endl;

    auto prediction_index = std::distance(result_ptr, best_prediction);
    std::cout << "prediction_index: " << prediction_index << std::endl;

    auto prediction_label = labels.at(prediction_index);
    std::cout << "prediction_label: " << prediction_label << std::endl;
}

int main(int argc, char** argv)
{
    std::string instanceName{ "image-classification-inference" };
    std::string imageFilepath{ "../../../data/images/dog.jpg" };
    std::string labelFilepath{ "../../../data/labels/synset.txt" };
    std::string modelFilepath{ "../../../data/models/squeezenet1.1-7.onnx" };

    std::vector<std::string> labels{ readLabels(labelFilepath) };

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);

    // OrtCUDAProviderOptions cuda_options;
    // sessionOptions.AppendExecutionProvider_CUDA(cuda_options);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    const char* inputName = session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType << std::endl;

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    const char* outputName = session.GetOutputName(0, allocator);
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::cout << "Output Type: " << outputType << std::endl;

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;

    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR, cv::Size(inputDims.at(2), inputDims.at(3)), cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resizedImage);
    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
    auto input_bytes = preprocessedImage.total() * preprocessedImage.elemSize();

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(Ort::Value::CreateTensor(memoryInfo, preprocessedImage.data, input_bytes, inputDims.data(), inputDims.size(), inputType));

    std::vector<const char*> inputNames{ inputName };
    std::vector<const char*> outputNames{ outputName };
    std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), inputTensors.data(), 1, outputNames.data(), 1);

    auto print_results = [&](auto* result_ptr, unsigned long result_size) -> void
    {
        auto best_prediction = std::max_element(result_ptr, result_ptr + result_size);
        std::cout << "\n\nprediction_score: " << *best_prediction << std::endl;

        auto prediction_index = std::distance(result_ptr, best_prediction);
        std::cout << "prediction_index: " << prediction_index << std::endl;

        auto prediction_label = labels.at(prediction_index);
        std::cout << "prediction_label: " << prediction_label << std::endl;
    };

    for (auto&& output : outputTensors)
    {
        auto out_type = output.GetTensorTypeAndShapeInfo().GetElementType();
        auto out_size = output.GetTensorTypeAndShapeInfo().GetElementCount();

        switch (out_type)
        {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            {
                // auto out_data = output.GetTensorData<float>();
                parse_output<float>(output, out_size, labels);
                break;
            }

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            {
                // auto out_data = output.GetTensorData<uint8_t>();
                parse_output<uint8_t>(output, out_size, labels);
                break;
            }

            default:
                break;
        }
    }

    return 0;
}