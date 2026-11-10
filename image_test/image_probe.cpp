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

struct ImageMapping {
    cv::Size original_size;
    float scale_x;
    float scale_y;
    int pad_left;
    int pad_top;
};

cv::Mat prepare_input(const cv::Mat& image, bool letterbox, ImageMapping& mapping)
{
    cv::Size resized_size(kInputSize, kInputSize);
    if (letterbox) {
        const float scale = static_cast<float>(kInputSize) / std::max(image.cols, image.rows);
        resized_size.width = std::max(1, static_cast<int>(std::round(image.cols * scale)));
        resized_size.height = std::max(1, static_cast<int>(std::round(image.rows * scale)));
    }

    const int pad_x = kInputSize - resized_size.width;
    const int pad_y = kInputSize - resized_size.height;
    mapping = {image.size(), static_cast<float>(image.cols) / resized_size.width,
               static_cast<float>(image.rows) / resized_size.height, pad_x / 2, pad_y / 2};

    cv::Mat resized;
    cv::resize(image, resized, resized_size);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    if (pad_x == 0 && pad_y == 0) {
        return rgb;
    }

    // 记录左侧和顶部的填边，解码时先扣掉它们，再还原到原图。
    cv::Mat padded;
    cv::copyMakeBorder(rgb, padded, mapping.pad_top, pad_y - mapping.pad_top,
                       mapping.pad_left, pad_x - mapping.pad_left,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return padded;
}

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
                  const ImageMapping& mapping, bool scores_are_probabilities,
                  std::vector<Detection>& candidates)
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
    const auto* class_values = static_cast<const float*>(classes.buf);
    const cv::Rect2f image_bounds(0.0f, 0.0f,
                                 static_cast<float>(mapping.original_size.width),
                                 static_cast<float>(mapping.original_size.height));

    for (int cell = 0; cell < cells; ++cell) {
        // 输出是 NCHW，同一网格在相邻类别通道中的位置相隔 cells 个元素。
        int best_class = 0;
        float best_value = class_values[cell];
        for (int class_id = 1; class_id < class_count; ++class_id) {
            const float value = class_values[class_id * cells + cell];
            if (value > best_value) {
                best_class = class_id;
                best_value = value;
            }
        }

        // Model Zoo 已在网络内做了 sigmoid。旧六路模型输出的是原始 logits。
        const float score = scores_are_probabilities ? best_value
                                                     : 1.0f / (1.0f + std::exp(-best_value));
        if (score <= kScoreThreshold) {
            continue;
        }

        std::array<float, 4> distances{};
        for (std::size_t side = 0; side < distances.size(); ++side) {
            distances[side] = decode_distance(box_logits + side * kDflBins * cells + cell, cells);
        }

        const float center_x = cell % grid + 0.5f;
        const float center_y = cell / grid + 0.5f;
        const float left = ((center_x - distances[0]) * stride - mapping.pad_left) * mapping.scale_x;
        const float top = ((center_y - distances[1]) * stride - mapping.pad_top) * mapping.scale_y;
        const float right = ((center_x + distances[2]) * stride - mapping.pad_left) * mapping.scale_x;
        const float bottom = ((center_y + distances[3]) * stride - mapping.pad_top) * mapping.scale_y;
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

    rknn_context context{};
    int ret = rknn_init(&context, model.data(),
                        static_cast<std::uint32_t>(model.size()), 0, nullptr);
    std::cout << "rknn_init = " << ret << '\n';
    if (ret != 0) {
        return 1;
    }

    constexpr std::size_t kLegacyOutputCount = kStrides.size() * 2;
    constexpr std::size_t kModelZooOutputCount = kStrides.size() * 3;
    std::array<rknn_output, kModelZooOutputCount> outputs{};
    std::vector<Detection> candidates;
    rknn_input_output_num io{};
    ret = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (ret == 0) {
        std::cout << "Inputs: " << io.n_input << ", outputs: " << io.n_output << '\n';
        if (io.n_input != 1 || (io.n_output != kLegacyOutputCount && io.n_output != kModelZooOutputCount)) {
            std::cerr << "Expected one input and either six or nine YOLOv8 outputs\n";
            ret = -1;
        }
    } else {
        std::cerr << "rknn_query = " << ret << '\n';
    }

    const bool model_zoo_layout = io.n_output == kModelZooOutputCount;
    ImageMapping mapping{};
    cv::Mat input_image;
    if (ret == 0) {
        input_image = prepare_input(image, model_zoo_layout, mapping);
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
            // Model Zoo 每个尺度还有一个类别分数和张量，用于快速过滤。
            // 单图测试逐个检查类别分数，因此只需解码每组的前两个输出。
            const std::size_t outputs_per_scale = io.n_output / kStrides.size();
            for (std::size_t scale = 0; scale < kStrides.size(); ++scale) {
                const std::size_t offset = scale * outputs_per_scale;
                if (!decode_scale(outputs[offset], outputs[offset + 1], kStrides[scale],
                                  mapping, model_zoo_layout, candidates)) {
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
