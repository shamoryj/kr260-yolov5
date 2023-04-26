#include <cstdlib>
#include <cstring>
#include <random>
#include "Server.hpp"
#include "YoloModel.hpp"

#include "message.pb.h"

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

std::vector<Image> get_camera_images(const MyMessage& request) {
  // TODO: Implement camera control here
  static RandomGenerator rng;

  // Load images
  std::vector<Image> images = YoloModel::load_images("~/code/scenes");
  unsigned int rand_img_idx = rng.next_in_range(0, images.size() - 1);
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
  YoloModel model("~/code/quant_comp_v5m");

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
    std::vector<ImageResult> img_results = model.run_images(images);

    // Process results
    model.process_results(img_results, true, true);

    // Send results to host
    MyMessage reply;
    build_reply(request, reply, img_results);
    serv.send_message(reply);
  }

  return EXIT_SUCCESS;
}
