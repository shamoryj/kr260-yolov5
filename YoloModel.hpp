#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vitis/ai/yolov3.hpp>

class Timer {
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  std::chrono::duration<float> duration;

 public:
  void Start() { start = std::chrono::high_resolution_clock::now(); }

  void Stop() {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
  }

  float GetDurationInSeconds() const { return duration.count(); }

  long long GetDurationInMilliseconds() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
        .count();
  }

  void Reset() {
    start = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<float>(0);
  }

  void PrintDuration() const {
    std::cout << "Time taken: " << duration.count() << " seconds." << std::endl;
  }
};

struct Image {
  cv::Mat mat;
  std::filesystem::path path;

  Image(cv::Mat& img, std::filesystem::path& path) {
    this->mat = img;
    this->path = path;
  }

  Image(const Image& other) {
    this->mat = other.mat.clone();
    this->path = other.path;
  }
};

struct DetectedObject {
  std::string label;
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float confidence;

  DetectedObject(const vitis::ai::YOLOv3Result::BoundingBox& box,
                 const cv::Mat& img,
                 const std::vector<std::string>& class_labels) {
    label = class_labels[box.label];
    xmin = box.x * img.cols + 1;
    ymin = box.y * img.rows + 1;
    xmax = xmin + box.width * img.cols;
    ymax = ymin + box.height * img.rows;
    confidence = box.score;

    // Clamp coordinates to the image boundaries
    if (xmin < 0.f) xmin = 1.f;
    if (ymin < 0.f) ymin = 1.f;
    if (xmax > img.cols) xmax = img.cols;
    if (ymax > img.rows) ymax = img.rows;
  }
};

struct ImageResult {
  Image img;
  Image bbox_img;
  std::vector<DetectedObject> objs;

  ImageResult(
      const Image& img,
      const std::vector<vitis::ai::YOLOv3Result::BoundingBox>& img_bboxes,
      const std::vector<std::string>& class_labels)
      : img(img), bbox_img(img) {
    for (auto& box : img_bboxes) {
      objs.emplace_back(box, img.mat, class_labels);
    }
  }
};

class YoloModel {
 public:
  static std::vector<Image> load_images(const std::string& path);

  explicit YoloModel(const std::string& model_path);
  std::vector<ImageResult> run_images(std::vector<Image>& images);
  void process_results(std::vector<ImageResult>& img_results,
                       bool print_results, bool save_img);

 private:
  static bool is_image_file(const std::filesystem::path& path);
  static std::filesystem::path get_absolute_path(const std::string& path);
  std::vector<std::string> get_classes(
      const std::filesystem::path& prototxt_path);
  void draw_bounding_box(cv::Mat& img, DetectedObject& obj);

  std::unique_ptr<vitis::ai::YOLOv3> model{};
  std::vector<std::string> class_labels;
};