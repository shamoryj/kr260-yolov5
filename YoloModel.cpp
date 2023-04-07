#include "YoloModel.hpp"

std::vector<Image> YoloModel::load_images(const std::string& path) {
  std::vector<Image> images;

  // Convert relative path to absolute path
  std::filesystem::path abs_path = get_absolute_path(path);
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

YoloModel::YoloModel(const std::string& path) {
  std::filesystem::path model_path = get_absolute_path(path);
  if (std::filesystem::is_directory(model_path)) {
    std::string name = model_path.stem().string();
    std::filesystem::path prototxt_path = model_path / (name + ".prototxt");
    std::filesystem::path xmodel_path = model_path / (name + ".xmodel");

    if (std::filesystem::exists(prototxt_path) &&
        std::filesystem::exists(xmodel_path)) {
      std::filesystem::path current_path =
          std::filesystem::current_path();  // get current directory path
      std::filesystem::path new_dir_path =
          current_path / name;  // create new directory path

      try {
        std::filesystem::create_directory(
            new_dir_path);  // create new directory
        std::filesystem::copy(
            prototxt_path,
            new_dir_path);  // copy .prototxt file to new directory
        std::filesystem::copy(
            xmodel_path, new_dir_path);  // copy .xmodel file to new directory

        this->model = vitis::ai::YOLOv3::create(name, true);
        this->class_labels = get_classes(prototxt_path);
      } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
      }
    } else {
      std::cerr << "Model path directory does not contain .prototxt and/or "
                   ".xmodel files with the same name."
                << std::endl;
      std::cerr << "For .prototxt details, see: "
                   "https://docs.xilinx.com/r/en-US/ug1354-xilinx-ai-sdk/"
                   "Using-the-Configuration-File"
                << std::endl;
    }
  } else {
    std::cerr << "Model path is not a directory." << std::endl;
  }
}

std::vector<ImageResult> YoloModel::run_images(
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
    auto results = model->run(img.mat);
    t.Stop();
    std::cout << "Completed " << img.path.filename() << " in "
              << t.GetDurationInMilliseconds() << " milliseconds!" << std::endl;
    total_duration += t.GetDurationInMilliseconds();
    img_results.emplace_back(img, results.bboxes, class_labels);
  }

  std::cout << std::endl
            << "Completed " << images.size() << " image(s) in "
            << total_duration << " milliseconds!" << std::endl;
  std::cout << "Average time: " << total_duration / images.size() << " ms"
            << std::endl;

  return img_results;
}

void YoloModel::process_results(std::vector<ImageResult>& img_results,
                                bool print_results, bool save_img) {
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
      draw_bounding_box(img_result.bbox_img.mat, obj);
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
        img_result.bbox_img.path =
            save_img_dir / img_result.img.path.filename();
        if (cv::imwrite(img_result.bbox_img.path, img_result.bbox_img.mat)) {
          std::cout << std::endl
                    << "Result image saved to: " << img_result.bbox_img.path
                    << std::endl;
        } else {
          std::cout << std::endl
                    << "Failed to save result image to: "
                    << img_result.bbox_img.path << std::endl;
        }
      }
    }
  }
}

bool YoloModel::is_image_file(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::filesystem::path YoloModel::get_absolute_path(const std::string& path) {
  // Check if the path starts with ~
  if (!path.empty() && path[0] == '~') {
    // Get the value of the HOME environment variable
    char* home = std::getenv("HOME");

    // Replace the ~ with the value of the HOME environment variable
    return home + path.substr(1);
  } else {
    return std::filesystem::absolute(path);
  }
}

std::vector<std::string> YoloModel::get_classes(
    const std::filesystem::path& prototxt_path) {
  std::ifstream file(prototxt_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open file " << prototxt_path << std::endl;
    return {};
  }

  std::vector<std::string> classes;
  std::string line;
  while (std::getline(file, line)) {
    // remove leading whitespace
    line.erase(0, line.find_first_not_of(" \t"));

    if (line.compare(0, 8, "classes:") == 0) {
      // extract class label string between quotes
      std::string label = line.substr(line.find('"') + 1);
      label = label.substr(0, label.find('"'));
      classes.push_back(label);
    }
  }
  file.close();

  return classes;
}

void YoloModel::draw_bounding_box(cv::Mat& img, DetectedObject& obj) {
  // Define BGR encoded colors for bounding box and text
  auto red = cv::Scalar(0, 0, 255);
  auto white = cv::Scalar(255, 255, 255);

  // Draw the bounding box around the detected object
  cv::rectangle(img, cv::Point(obj.xmin, obj.ymin),
                cv::Point(obj.xmax, obj.ymax), red, 2, 1, 0);

  // Prepare confidence text and its properties
  std::string conf_text = obj.label + " " + std::to_string(obj.confidence);
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
