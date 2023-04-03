#include <algorithm>
#include <chrono>
#include <filesystem>
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
};

struct DetectedObject {
  int label;
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float confidence;

  DetectedObject(const vitis::ai::YOLOv3Result::BoundingBox& box,
                 const cv::Mat& img) {
    label = box.label;
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
  std::vector<DetectedObject> objs;

  ImageResult(
      Image& img,
      const std::vector<vitis::ai::YOLOv3Result::BoundingBox>& img_bboxes)
      : img(img) {
    for (auto& box : img_bboxes) {
      objs.emplace_back(box, img.mat);
    }
  }
};

bool is_image_file(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<Image> load_images(const std::string& path) {
  std::vector<Image> images;

  // Convert relative path to absolute path
  std::filesystem::path abs_path;
  // Check if the path starts with ~
  if (!path.empty() && path[0] == '~') {
    // Get the value of the HOME environment variable
    char* home = std::getenv("HOME");

    // Replace the ~ with the value of the HOME environment variable
    abs_path = home + path.substr(1);
  } else {
    abs_path = std::filesystem::absolute(path);
  }
  std::cout << std::endl << "Absolute path: " << abs_path << std::endl;

  // Check if path is a directory
  if (std::filesystem::is_directory(abs_path)) {
    std::cout << "Path points to a directory." << std::endl;

    // Iterate through all items in the directory
    for (const auto& entry : std::filesystem::directory_iterator(abs_path)) {
      std::filesystem::path entry_path = entry.path();
      if (is_image_file(entry_path)) {
        std::cout << "Image file: " << entry_path << std::endl;

        // Load the image using OpenCV
        cv::Mat img = cv::imread(entry_path.string());

        // Add the image to the vector of images
        if (!img.empty()) {
          images.emplace_back(img, entry_path);
        } else {
          std::cout << "Failed to load: " << entry_path.filename() << std::endl;
        }
      }
    }
  } else if (std::filesystem::is_regular_file(abs_path)) {
    std::cout << "Path points to a file." << std::endl;
    // Load the image using OpenCV
    cv::Mat img = cv::imread(abs_path.string());

    // Add the image to the vector of images
    if (!img.empty()) {
      images.emplace_back(img, abs_path);
    } else {
      std::cout << "Failed to load: " << abs_path.filename() << std::endl;
    }
  } else {
    std::cout << "Path does not exist or is not a file or directory."
              << std::endl;
  }

  std::cout << "Loaded " << images.size() << " image(s)." << std::endl;

  return images;
}

std::vector<ImageResult> run_images(std::unique_ptr<vitis::ai::YOLOv3>& yolo,
                                    std::vector<Image>& images) {
  std::vector<ImageResult> img_results;
  Timer t;
  long long total_duration = 0;

  std::cout << std::endl
            << "Running " << images.size() << " image(s)." << std::endl;

  for (auto& img : images) {
    // Run the YOLO model and get the results
    std::cout << std::endl
              << "Running " << img.path.filename() << "..." << std::endl;
    t.Start();
    auto results = yolo->run(img.mat);
    t.Stop();
    std::cout << "Completed " << img.path.filename() << " in "
              << t.GetDurationInMilliseconds() << " milliseconds!" << std::endl;
    total_duration += t.GetDurationInMilliseconds();
    img_results.emplace_back(img, results.bboxes);
  }

  std::cout << std::endl
            << "Completed " << images.size() << " image(s) in "
            << total_duration << " milliseconds!" << std::endl;
  std::cout << "Average time: " << total_duration / images.size() << " ms"
            << std::endl;

  return img_results;
}

void draw_bounding_box(cv::Mat& img, DetectedObject& obj) {
  // Define BGR encoded colors for bounding box and text
  auto red = cv::Scalar(0, 0, 255);
  auto white = cv::Scalar(255, 255, 255);

  // Draw the bounding box around the detected object
  cv::rectangle(img, cv::Point(obj.xmin, obj.ymin),
                cv::Point(obj.xmax, obj.ymax), red, 2, 1, 0);

  // Prepare confidence text and its properties
  std::string conf_text = std::to_string(obj.confidence);
  int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
  double font_scale = 1.5;
  int font_thickness = 2;
  int baseline;
  cv::Size text_size = cv::getTextSize(conf_text, font_face, font_scale,
                                       font_thickness, &baseline);

  // Calculate text position and handle edge cases
  int padding = 3;
  int text_x = obj.xmin;
  if (text_x + text_size.width > img.cols) {
    text_x = obj.xmax - text_size.width;
  }
  int text_y = obj.ymin - padding;
  if (text_y - text_size.height < 0) {
    text_y = obj.ymax + padding + text_size.height - 1;
  }

  // Draw a red background rectangle for the text
  cv::rectangle(img, cv::Point(text_x, text_y - text_size.height),
                cv::Point(text_x + text_size.width, text_y + padding), red, -1);

  // Draw the confidence text in white
  cv::putText(img, conf_text, cv::Point(text_x, text_y), font_face, font_scale,
              white, font_thickness);
}

void process_results(std::vector<ImageResult>& img_results, bool print_results,
                     bool save_img) {
  for (auto& img_result : img_results) {
    // Iterate through the detected bounding boxes
    for (auto& obj : img_result.objs) {
      if (print_results) {
        if (&obj == &img_result.objs.front()) {
          // Add blank line before first result
          std::cout << std::endl;
        }
        std::cout << "RESULT: " << obj.label << "\t" << obj.xmin << "\t"
                  << obj.ymin << "\t" << obj.xmax << "\t" << obj.ymax << "\t"
                  << obj.confidence << std::endl;
      }
      if (save_img) {
        draw_bounding_box(img_result.img.mat, obj);
      }
    }

    // Save the output image
    if (save_img) {
      bool success = true;
      std::filesystem::path save_img_dir =
          img_result.img.path.parent_path() / "results";

      // Create save img directory if it does not exist
      if (!std::filesystem::exists(save_img_dir)) {
        try {
          std::filesystem::create_directory(save_img_dir);
        } catch (const std::exception& ex) {
          std::cerr << "Failed to create directory: " << ex.what() << std::endl;
          success = false;
        }
      }

      // Attempt to save image
      if (success) {
        std::string save_img_path =
            save_img_dir / img_result.img.path.filename();
        if (cv::imwrite(save_img_path, img_result.img.mat)) {
          std::cout << std::endl
                    << "Result image saved to: " << save_img_path << std::endl;
        } else {
          std::cout << std::endl
                    << "Failed to save result image to: " << save_img_path
                    << std::endl;
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  std::string images_path = "~/code/shiprs_test_images";
  if (argc > 1) {
    images_path = argv[1];
  } else {
    std::cout << std::endl
              << "Using default image path: " << images_path
              << std::endl;
  }

  // Load YOLO model
  auto yolo = vitis::ai::YOLOv3::create("quant_comp_v5m", true);

  // Load images
  std::vector<Image> images = load_images(images_path);

  // Run images
  auto img_results = run_images(yolo, images);

  // Process results
  process_results(img_results, true, true);

  return EXIT_SUCCESS;
}