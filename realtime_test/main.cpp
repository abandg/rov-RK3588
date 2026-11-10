#include "rknn_api.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
constexpr std::array<int, 3> kStrides{{8, 16, 32}};
constexpr std::uint32_t kLegacyOutputCount = kStrides.size() * 2;
constexpr std::uint32_t kModelZooOutputCount = kStrides.size() * 3;

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
        // 扣掉输入填边后，再把坐标缩放回摄像头原始画面。
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
    const cv::Scalar color(0, 255, 0);
    for (const Detection& detection : detections) {
        const cv::Rect box = detection.box;
        const auto label = cv::format("%s %.2f", kClassNames[detection.class_id], detection.score);
        cv::rectangle(image, box, color, 2);
        cv::putText(image, label, cv::Point(box.x, std::max(12, box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }
}

bool check_rknn(const char* operation, int result)
{
    if (result == 0) {
        return true;
    }
    std::cerr << operation << " failed: " << result << '\n';
    return false;
}

struct FrameOutputs {
    FrameOutputs(rknn_context handle, std::uint32_t output_count)
        : context(handle), count(output_count) {}
    FrameOutputs(const FrameOutputs&) = delete;
    FrameOutputs& operator=(const FrameOutputs&) = delete;

    ~FrameOutputs()
    {
        if (acquired) {
            rknn_outputs_release(context, count, values.data());
        }
    }

    std::array<rknn_output, kModelZooOutputCount> values{};
    rknn_context context;
    std::uint32_t count;
    bool acquired = false;
};

class Detector {
public:
    Detector() = default;
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    ~Detector()
    {
        if (initialized_) {
            rknn_destroy(context_);
        }
    }

    bool load(const char* model_path)
    {
        std::ifstream file(model_path, std::ios::binary);
        model_.assign(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
        if (model_.empty()) {
            std::cerr << "Could not read model: " << model_path << '\n';
            return false;
        }

        const int result = rknn_init(&context_, model_.data(),
                                     static_cast<std::uint32_t>(model_.size()), 0, nullptr);
        if (!check_rknn("rknn_init", result)) {
            return false;
        }
        initialized_ = true;

        rknn_input_output_num io{};
        if (!check_rknn("rknn_query", rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)))) {
            return false;
        }
        if (io.n_input != 1 || (io.n_output != kLegacyOutputCount && io.n_output != kModelZooOutputCount)) {
            std::cerr << "Expected one input and either six or nine YOLOv8 outputs\n";
            return false;
        }

        output_count_ = io.n_output;
        std::cout << "Loaded model: " << model_path << " (" << output_count_ << " outputs)\n";
        return true;
    }

    bool process(cv::Mat& frame)
    {
        const bool model_zoo_layout = output_count_ == kModelZooOutputCount;
        ImageMapping mapping{};
        const cv::Mat& input_image = prepare_input(frame, model_zoo_layout, mapping);

        rknn_input input{};
        input.buf = input_image.data;
        input.size = static_cast<std::uint32_t>(input_image.total() * input_image.elemSize());
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        if (!check_rknn("rknn_inputs_set", rknn_inputs_set(context_, 1, &input)) ||
            !check_rknn("rknn_run", rknn_run(context_, nullptr))) {
            return false;
        }

        std::vector<Detection> candidates;
        {
            // 每帧单独获取并释放输出，模型上下文在整个预览过程中复用。
            FrameOutputs outputs(context_, output_count_);
            for (std::uint32_t i = 0; i < output_count_; ++i) {
                outputs.values[i].index = i;
                outputs.values[i].want_float = 1;
            }
            outputs.acquired = check_rknn(
                "rknn_outputs_get", rknn_outputs_get(context_, output_count_, outputs.values.data(), nullptr));
            if (!outputs.acquired) {
                return false;
            }
            const std::size_t outputs_per_scale = output_count_ / kStrides.size();
            for (std::size_t scale = 0; scale < kStrides.size(); ++scale) {
                const std::size_t offset = scale * outputs_per_scale;
                if (!decode_scale(outputs.values[offset], outputs.values[offset + 1],
                                  kStrides[scale], mapping, model_zoo_layout, candidates)) {
                    return false;
                }
            }
        }

        draw_detections(frame, suppress_overlaps(candidates));
        return true;
    }

private:
    const cv::Mat& prepare_input(const cv::Mat& frame, bool letterbox, ImageMapping& mapping)
    {
        cv::Size resized_size(kInputSize, kInputSize);
        if (letterbox) {
            const float scale = static_cast<float>(kInputSize) / std::max(frame.cols, frame.rows);
            resized_size.width = std::max(1, static_cast<int>(std::round(frame.cols * scale)));
            resized_size.height = std::max(1, static_cast<int>(std::round(frame.rows * scale)));
        }

        const int pad_x = kInputSize - resized_size.width;
        const int pad_y = kInputSize - resized_size.height;
        mapping = {frame.size(), static_cast<float>(frame.cols) / resized_size.width,
                   static_cast<float>(frame.rows) / resized_size.height, pad_x / 2, pad_y / 2};

        // 在独立缓冲区中预处理，让显示和画框继续使用原始 BGR 帧。
        cv::resize(frame, resized_, resized_size);
        cv::cvtColor(resized_, rgb_, cv::COLOR_BGR2RGB);
        if (pad_x == 0 && pad_y == 0) {
            return rgb_;
        }
        cv::copyMakeBorder(rgb_, padded_, mapping.pad_top, pad_y - mapping.pad_top,
                           mapping.pad_left, pad_x - mapping.pad_left,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return padded_;
    }

    std::vector<char> model_;
    cv::Mat resized_;
    cv::Mat rgb_;
    cv::Mat padded_;
    rknn_context context_{};
    std::uint32_t output_count_ = 0;
    bool initialized_ = false;
};

}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " MODEL.rknn /dev/video0\n";
        return 1;
    }

    const char* window = "RK3588 YOLOv8";
    try {
        cv::VideoCapture camera(argv[2], cv::CAP_V4L2);
        if (!camera.isOpened()) {
            std::cerr << "Failed to open V4L2 device " << argv[2] << '\n';
            return 1;
        }

        Detector detector;
        if (!detector.load(argv[1])) {
            return 1;
        }

        std::cout << "Press Q or Esc in the preview window to exit\n";
        cv::Mat frame;
        for (;;) {
            if (!camera.read(frame) || frame.empty()) {
                std::cerr << "Could not read a frame from " << argv[2] << '\n';
                return 1;
            }
            if (!detector.process(frame)) {
                return 1;
            }

            cv::imshow(window, frame);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                break;
            }
        }
        cv::destroyWindow(window);
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
