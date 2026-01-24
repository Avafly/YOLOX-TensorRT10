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
        std::printf("Usage: %s model image [target size] [conf] [nms]\n", argv[0]);
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

    // load input
    cv::Mat image_host = cv::imread(argv[2], cv::IMREAD_COLOR_BGR);
    if (image_host.empty())
    {
        std::cout << "Failed to read image\n";
        return -1;
    }

    // create detector
    Infer::YOLOXDetector detector{
        argv[1], conf_thres, nms_thres, target_size, static_cast<int>(labels.size())
    };
    if (!detector.IsInited())
    {
        std::cout << "Faield to create detector\n";
        return -1;
    }

    // timer
    int64 start_time = cv::getTickCount();

    auto objects = detector.Detect(image_host);

    // show elapsed time
    std::printf("Elapsed time: %.1fms\n", (cv::getTickCount() - start_time) / cv::getTickFrequency() * 1000.0);

    // visualize result
    if (Infer::YOLOXDetector::DrawObjects(image_host, objects, labels, true))
        cv::imwrite("results.png", image_host);
    else
        std::cout << "Failed to draw objects\n";

    return 0;
}
