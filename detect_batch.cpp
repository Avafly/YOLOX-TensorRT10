#include <iostream>
#include <vector>
#include <string>
#include <cstdio>

#include <opencv2/opencv.hpp>
#include "yolox_detector.h"

const std::vector<std::string> labels = {
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train",
    "Truck", "Boat", "Traffic light", "Fire hydrant", "Stop sign", "Parking meter",
    "Bench", "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear",
    "Zebra", "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase",
    "Frisbee", "Skis", "Snowboard", "Sports ball", "Kite", "Baseball bat",
    "Baseball glove", "Skateboard", "Surfboard", "Tennis racket", "Bottle",
    "Wine glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple",
    "Sandwich", "Orange", "Broccoli", "Carrot", "Hot dog", "Pizza", "Donut",
    "Cake", "Chair", "Couch", "Potted plant", "Bed", "Dining table", "Toilet",
    "Tv", "Laptop", "Mouse", "Remote", "Keyboard", "Cell phone", "Microwave",
    "Oven", "Toaster", "Sink", "Refrigerator", "Book", "Clock", "Vase", "Scissors",
    "Teddy bear", "Hair drier", "Toothbrush"
};

int main(int argc, char *argv[])
{
    // settings
    if (argc < 3)
    {
        std::printf("Usage: %s model folder [target size] [conf] [nms]\n", argv[0]);
        return 0;
    }
    int target_size = 640;
    float conf_thres = 0.25f, nms_thres = 0.45f;
    if (argc >= 4 && std::stoi(argv[3]) > 0)
        target_size = std::stoi(argv[3]);
    if (argc >= 5 && std::stof(argv[4]) > 0.0f && std::stof(argv[4]) <= 1.0f)
        conf_thres = std::stof(argv[4]);
    if (argc >= 6 && std::stof(argv[5]) > 0.0f && std::stof(argv[5]) <= 1.0f)
        nms_thres = std::stof(argv[5]);
    // show settings
    std::cout << "Model:       " << argv[1] << "\n";
    std::cout << "Input:       " << argv[2] << "\n";
    std::cout << "Target size: " << target_size << "\n";
    std::cout << "Conf:        " << conf_thres << "\n";
    std::cout << "NMS:         " << nms_thres << "\n";

    // load image paths from folder
    std::vector<cv::String> image_paths;
    cv::glob(argv[2], image_paths);
    if (image_paths.empty())
    {
        std::cout << "No images found in " << argv[2] << "\n";
        return -1;
    }
    std::cout << "Found " << image_paths.size() << " files\n";

    // create detector
    Infer::YOLOXDetector detector{
        argv[1], conf_thres, nms_thres, target_size, static_cast<int>(labels.size())
    };
    if (!detector.IsInited())
    {
        std::cout << "Failed to create detector\n";
        return -1;
    }

    const int max_batch_size = detector.GetMaxBatchSize();
    std::cout << "Max batch size: " << max_batch_size << "\n";

    double total_infer_time = 0.0;
    int saved = 0;

    for (size_t i = 0; i < image_paths.size(); i += max_batch_size)
    {
        int current_batch = std::min(max_batch_size, static_cast<int>(image_paths.size() - i));

        // load batch images
        std::vector<cv::Mat> batch_images;
        batch_images.reserve(current_batch);
        for (int j = 0; j < current_batch; ++j)
        {
            cv::Mat image = cv::imread(image_paths[i + j], cv::IMREAD_COLOR_BGR);
            if (image.empty())
            {
                std::cerr << "Failed to read " << image_paths[i + j] << "\n";
                continue;
            }
            batch_images.push_back(std::move(image));
        }
        if (batch_images.empty())
            continue;

        // inference
        int64 start_time = cv::getTickCount();
        auto batch_objects = detector.Detect(batch_images);
        total_infer_time += (cv::getTickCount() - start_time) / cv::getTickFrequency() * 1000.0;

        // draw and save
        for (size_t j = 0; j < batch_images.size(); ++j)
        {
            Infer::YOLOXDetector::DrawObjects(batch_images[j], batch_objects[j], labels, true);
            cv::imwrite("results_" + std::to_string(saved++) + ".png", batch_images[j]);
        }
    }

    std::printf("Inference time: %.1fms\n", total_infer_time);
    std::cout << "Saved " << saved << " results\n";

    return 0;
}
