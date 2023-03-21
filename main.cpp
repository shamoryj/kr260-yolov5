#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <string>
#include <vitis/ai/yolov3.hpp>

#include "message.pb.h"

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

struct RandomGenerator {
  // Mersenne Twister engine with a 32-bit state size
  std::mt19937_64 mt_engine;

  // Constructor that seeds the engine with the current time
  RandomGenerator() : mt_engine(std::random_device()()) {}

  // Function to generate a random number
  unsigned int next() { return static_cast<unsigned int>(mt_engine()); }

  // Function to generate a random number in the range [start, end]
  unsigned int next_in_range(unsigned int start, unsigned int end) {
    return start + (next() % (end - start + 1));
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
  Image bbox_img;
  std::vector<DetectedObject> objs;

  ImageResult(
      const Image& img,
      const std::vector<vitis::ai::YOLOv3Result::BoundingBox>& img_bboxes)
      : img(img), bbox_img(img) {
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

double secondsSinceEpoch() {
  // get the current time
  std::time_t current_time = std::time(nullptr);

  // convert to seconds since the epoch
  return std::difftime(current_time, 0);
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

struct Server {
 private:
  int listenSockfd = -1;
  short port;
  int sockfd = -1;

 public:
  explicit Server(short port) : port(port) {}
  ~Server() {
    // Close the socket and listening socket
    close(sockfd);
    close(listenSockfd);
  }
  bool start() {
    if (listenSockfd != -1) {
      std::cerr << "Error: Server already started" << std::endl;
      return true;
    }
    // Create a TCP socket to listen for incoming connections
    listenSockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenSockfd == -1) {
      std::cerr << "Error: Failed to create socket" << std::endl;
      return false;
    }

    // Set the SO_REUSEADDR option to allow reuse of the same address and port
    int reuseaddr = 1;
    if (setsockopt(listenSockfd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr,
                   sizeof(reuseaddr)) == -1) {
      std::cerr << "Error: Failed to set SO_REUSEADDR option" << std::endl;
      return false;
    }

    // Bind the socket to a port
    struct sockaddr_in listenAddr {};
    memset(&listenAddr, 0, sizeof(listenAddr));
    listenAddr.sin_family = AF_INET;
    listenAddr.sin_addr.s_addr = INADDR_ANY;
    listenAddr.sin_port = htons(port);
    if (bind(listenSockfd, (struct sockaddr*)&listenAddr, sizeof(listenAddr)) ==
        -1) {
      std::cerr << "Error: Failed to bind socket" << std::endl;
      return false;
    }

    // Listen for incoming connections
    if (listen(listenSockfd, 5) == -1) {
      std::cerr << "Error: Failed to listen for incoming connections"
                << std::endl;
      return false;
    }

    return true;
  }

  bool accept_connection() {
    if (listenSockfd == -1) {
      std::cerr << "Error: Server is not running" << std::endl;
      return false;
    }
    // Accept an incoming connection
    struct sockaddr_in remoteAddr {};
    socklen_t remoteAddrLen = sizeof(remoteAddr);
    sockfd =
        accept(listenSockfd, (struct sockaddr*)&remoteAddr, &remoteAddrLen);
    if (sockfd == -1) {
      std::cerr << "Error: Failed to accept incoming connection" << std::endl;
      return false;
    }
    return true;
  }

  bool receive_message(MyMessage& message) {
    if (listenSockfd == -1) {
      std::cerr << "Error: Server is not running" << std::endl;
      return false;
    }
    if (sockfd == -1) {
      std::cerr << "Error: No established connection" << std::endl;
      return false;
    }
    // Read the message from the socket
    char buffer[1024];
    ssize_t numRecv = recv(sockfd, buffer, sizeof(buffer), 0);
    if (numRecv == -1) {
      std::cerr << "Error: Failed to receive message" << std::endl;
      return false;
    }

    // Parse the message from the byte array
    message.ParseFromArray(buffer, numRecv);
    return true;
  }

  bool send_message(MyMessage& message) {
    if (listenSockfd == -1) {
      std::cerr << "Error: Server is not running" << std::endl;
      return false;
    }
    if (sockfd == -1) {
      std::cerr << "Error: No established connection" << std::endl;
      return false;
    }
    // Serialize the message to a byte array
    message.set_time_sent(secondsSinceEpoch());
    size_t size = message.ByteSizeLong();
    char* messageData = (char*)malloc(size);
    message.SerializeToArray(messageData, size);

    // Send the message size to the socket
    ssize_t numSent = send(sockfd, &size, sizeof(size), 0);
    if (numSent == -1) {
      std::cerr << "Error: Failed to send message size" << std::endl;
      free(messageData);
      return false;
    }

    // Send the message data to the socket
    size_t bytesSent = 0;
    while (bytesSent < size) {
      numSent = send(sockfd, messageData + bytesSent, size - bytesSent, 0);
      if (numSent == -1) {
        std::cerr << "Error: Failed to send message" << std::endl;
        free(messageData);
        return false;
      }
      bytesSent += numSent;
    }

    free(messageData);
    return true;
  }

};

std::vector<Image> get_camera_images(const MyMessage& request) {
  // TODO: Implement camera control here
  static RandomGenerator rng;

  // Load images
  std::vector<Image> images = load_images("~/code/scenes");
  unsigned int rand_img_idx = rng.next_in_range(0, images.size());
  return std::vector<Image>(images.begin() + rand_img_idx,
                            images.begin() + rand_img_idx + 1);
}

void package_image(const cv::Mat& img_src, MyMessage::Image* img_dst) {
  // Copy the image data to the img_dst message
  img_dst->set_data(reinterpret_cast<const char*>(img_src.data),
                    img_src.total() * img_src.elemSize());
  img_dst->set_width(img_src.cols);
  img_dst->set_height(img_src.rows);
  img_dst->set_channels(img_src.channels());
}

void build_reply(const MyMessage& request, MyMessage& reply,
                 std::vector<ImageResult>& img_results) {
  if (request.command() == MyMessage::REQUEST) {
    reply.set_id(request.id());
    reply.set_command(MyMessage::REPLY);
    for (auto& result : img_results) {
      for (auto& bbox : result.objs) {
        MyMessage::Reply::BoundingBox* box =
            reply.mutable_reply()->add_bounding_boxes();
        box->set_label(std::to_string(bbox.label));
        box->set_x_min(bbox.xmin);
        box->set_y_min(bbox.ymin);
        box->set_x_max(bbox.xmax);
        box->set_y_max(bbox.ymax);
        box->set_confidence(bbox.confidence);
      }
      if (request.request().get_image()) {
        package_image(result.img.mat, reply.mutable_reply()->mutable_image());
      }
      if (request.request().get_bounding_box_image()) {
        package_image(result.bbox_img.mat,
                      reply.mutable_reply()->mutable_bounding_box_image());
      }
    }
  } else {
    std::cerr << "Error: Unsupported command" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  Server serv(12345);

  // Start the server
  if (!serv.start()) {
    return EXIT_FAILURE;
  }

  // Load YOLO model
  auto yolo = vitis::ai::YOLOv3::create("quant_comp_v5m", true);

  // Wait for the host to connect
  if (!serv.accept_connection()) {
    return EXIT_FAILURE;
  }
  while (true) {
    // Wait for request from host
    MyMessage request;
    serv.receive_message(request);

    // Get images from camera
    auto images = get_camera_images(request);

    // Run images
    auto img_results = run_images(yolo, images);

    // Process results
    process_results(img_results, true, true);

    // Send results to host
    MyMessage reply;
    build_reply(request, reply, img_results);
    serv.send_message(reply);
  }

  return EXIT_SUCCESS;
}
