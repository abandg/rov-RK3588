#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " /dev/video0\n";
        return 1;
    }

    const std::string device = argv[1];
    const char* window = "RK3588 USB Camera";

    try {
        // 显式使用 Linux 的 V4L2 后端，并把设备路径交给运行参数。
        cv::VideoCapture camera(device, cv::CAP_V4L2);
        if (!camera.isOpened()) {
            std::cerr << "Failed to open V4L2 device " << device << '\n';
            return 1;
        }

        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            std::cerr << "Opened " << device << " but could not read a frame\n";
            return 1;
        }

        std::cout << "First frame from " << device << ": "
                  << frame.cols << 'x' << frame.rows << '\n'
                  << "Press Q or Esc in the preview window to exit" << std::endl;

        for (;;) {
            // frame 就是下一步交给 YOLO 的图像。waitKey 也处理窗口事件。
            cv::imshow(window, frame);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                break;
            }

            if (!camera.read(frame) || frame.empty()) {
                std::cerr << "Stopped receiving frames from " << device << '\n';
                return 1;
            }
        }

        cv::destroyWindow(window);
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
