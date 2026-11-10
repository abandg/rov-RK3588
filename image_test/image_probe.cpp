#include "rknn_api.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {

constexpr int kInputSize = 640;
constexpr int kDflBins = 16;
constexpr int kBoxChannels = 4 * kDflBins;
constexpr float kScoreThreshold = 0.5f;
constexpr float kIouThreshold = 0.5f;
constexpr char kResultPath[] = "yolov8_result.jpg";
constexpr std::array<int, 3> kStrides{{8, 16, 32}};

// 类别索引按此模型的 COCO 顺序排列。
constexpr std::array<const char*, 80> kClassNames{{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
}};

struct Detection {
    cv::Rect2f box;
    int class_id;
    float score;
};

float decode_distance(const float* logits, int cells)
{
    // DFL 用 16 个分桶表示一条边的距离。先做 softmax，再取分桶索引的加权平均。
    float maximum = logits[0];
    for (int bin = 1; bin < kDflBins; ++bin) {
        maximum = std::max(maximum, logits[bin * cells]);
    }

    float sum = 0.0f;
    float weighted_sum = 0.0f;
    for (int bin = 0; bin < kDflBins; ++bin) {
        const float weight = std::exp(logits[bin * cells] - maximum);
        sum += weight;
        weighted_sum += bin * weight;
    }
    return weighted_sum / sum;
}

bool decode_scale(const rknn_output& boxes, const rknn_output& classes, int stride,
                  cv::Size image_size, std::vector<Detection>& candidates)
{
    const int grid = kInputSize / stride;
    const int cells = grid * grid;
    const int class_count = static_cast<int>(kClassNames.size());
    if (boxes.buf == nullptr || classes.buf == nullptr ||
        boxes.size != kBoxChannels * cells * sizeof(float) ||
        classes.size != class_count * cells * sizeof(float)) {
        std::cerr << "Unexpected output size at stride " << stride << '\n';
        return false;
    }

    const auto* box_logits = static_cast<const float*>(boxes.buf);
    const auto* class_logits = static_cast<const float*>(classes.buf);
    const float scale_x = static_cast<float>(image_size.width) / kInputSize;
    const float scale_y = static_cast<float>(image_size.height) / kInputSize;
    const cv::Rect2f image_bounds(0.0f, 0.0f,
                                 static_cast<float>(image_size.width),
                                 static_cast<float>(image_size.height));

    for (int cell = 0; cell < cells; ++cell) {
        // 输出是 NCHW，同一网格在相邻类别通道中的位置相隔 cells 个元素。
        int best_class = 0;
        float best_logit = class_logits[cell];
        for (int class_id = 1; class_id < class_count; ++class_id) {
            const float logit = class_logits[class_id * cells + cell];
            if (logit > best_logit) {
                best_class = class_id;
                best_logit = logit;
            }
        }

        const float score = 1.0f / (1.0f + std::exp(-best_logit));
        if (score <= kScoreThreshold) {
            continue;
        }

        std::array<float, 4> distances{};
        for (std::size_t side = 0; side < distances.size(); ++side) {
            distances[side] = decode_distance(box_logits + side * kDflBins * cells + cell, cells);
        }

        const float center_x = cell % grid + 0.5f;
        const float center_y = cell / grid + 0.5f;
        // 先从网格坐标恢复到模型尺寸，再分别按宽高映射回原图。
        const float left = (center_x - distances[0]) * stride * scale_x;
        const float top = (center_y - distances[1]) * stride * scale_y;
        const float right = (center_x + distances[2]) * stride * scale_x;
        const float bottom = (center_y + distances[3]) * stride * scale_y;
        const cv::Rect2f box = cv::Rect2f(left, top, right - left, bottom - top) & image_bounds;
        if (!box.empty()) {
            candidates.push_back({box, best_class, score});
        }
    }
    return true;
}

std::vector<Detection> suppress_overlaps(std::vector<Detection>& candidates)
{
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<Detection> kept;
    for (const Detection& candidate : candidates) {
        // 同一类别的框重叠过多时，保留分数较高的框。
        const bool duplicate = std::any_of(kept.begin(), kept.end(), [&](const Detection& other) {
            if (candidate.class_id != other.class_id) {
                return false;
            }
            const float intersection = (candidate.box & other.box).area();
            const float union_area = candidate.box.area() + other.box.area() - intersection;
            return intersection > kIouThreshold * union_area;
        });
        if (!duplicate) {
            kept.push_back(candidate);
        }
    }
    return kept;
}

void draw_detections(cv::Mat& image, const std::vector<Detection>& detections)
{
    std::cout << "Detections: " << detections.size() << '\n';
    const cv::Scalar color(0, 255, 0);
    for (const Detection& detection : detections) {
        const cv::Rect box = detection.box;
        const auto label = cv::format("%s %.2f", kClassNames[detection.class_id], detection.score);
        std::cout << label << '\n';
        cv::rectangle(image, box, color, 2);
        cv::putText(image, label, cv::Point(box.x, std::max(12, box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }
}

}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " MODEL.rknn IMAGE.jpg\n";
        return 1;
    }

    std::ifstream model_file(argv[1], std::ios::binary);
    std::vector<char> model(std::istreambuf_iterator<char>{model_file},
                            std::istreambuf_iterator<char>{});
    if (model.empty()) {
        std::cerr << "Could not read model: " << argv[1] << '\n';
        return 1;
    }

    cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not read image: " << argv[2] << '\n';
        return 1;
    }

    // 复现这个模型的输入预处理，将原图缩放为 640×640 RGB。
    cv::Mat input_image;
    cv::resize(image, input_image, cv::Size(kInputSize, kInputSize));
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

    rknn_context context{};
    int ret = rknn_init(&context, model.data(),
                        static_cast<std::uint32_t>(model.size()), 0, nullptr);
    std::cout << "rknn_init = " << ret << '\n';
    if (ret != 0) {
        return 1;
    }

    std::array<rknn_output, kStrides.size() * 2> outputs{};
    std::vector<Detection> candidates;
    rknn_input_output_num io{};
    ret = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (ret == 0) {
        std::cout << "Inputs: " << io.n_input << ", outputs: " << io.n_output << '\n';
        if (io.n_input != 1 || io.n_output != outputs.size()) {
            std::cerr << "Expected one input and six outputs for this YOLOv8 model\n";
            ret = -1;
        }
    } else {
        std::cerr << "rknn_query = " << ret << '\n';
    }

    if (ret == 0) {
        // 输入是交错排列的 RGB 字节，由 RKNN 运行库转换成模型所需格式。
        rknn_input input{};
        input.buf = input_image.data;
        input.size = static_cast<std::uint32_t>(input_image.total() * input_image.elemSize());
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        ret = rknn_inputs_set(context, 1, &input);
        std::cout << "rknn_inputs_set = " << ret << '\n';
    }

    if (ret == 0) {
        ret = rknn_run(context, nullptr);
        std::cout << "rknn_run = " << ret << '\n';
    }

    if (ret == 0) {
        for (std::uint32_t i = 0; i < io.n_output; ++i) {
            outputs[i].index = i;
            outputs[i].want_float = 1;
        }
        ret = rknn_outputs_get(context, io.n_output, outputs.data(), nullptr);
        std::cout << "rknn_outputs_get = " << ret << '\n';
        if (ret == 0) {
            for (std::uint32_t i = 0; i < io.n_output; ++i) {
                std::cout << "output[" << i << "]: " << outputs[i].size << " bytes\n";
            }
            // 每个尺度依次是位置和类别两个张量，共三对输出。
            for (std::size_t scale = 0; scale < kStrides.size(); ++scale) {
                if (!decode_scale(outputs[scale * 2], outputs[scale * 2 + 1],
                                  kStrides[scale], image.size(), candidates)) {
                    ret = -1;
                    break;
                }
            }
            // 输出缓冲区由运行库分配，使用完后通过同一个上下文释放。
            rknn_outputs_release(context, io.n_output, outputs.data());
        }
    }

    rknn_destroy(context);
    if (ret != 0) {
        return 1;
    }

    draw_detections(image, suppress_overlaps(candidates));
    if (!cv::imwrite(kResultPath, image)) {
        std::cerr << "Could not write " << kResultPath << '\n';
        return 1;
    }
    std::cout << "Saved " << kResultPath << '\n';
    return 0;
}
