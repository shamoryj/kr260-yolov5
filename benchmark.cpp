#include "YoloModel.hpp"

int main(int argc, char* argv[]) {
  std::string images_path = "~/code/shiprs_test_images";
  if (argc > 1) {
    images_path = argv[1];
  } else {
    std::cout << std::endl
              << "Using default image path: " << images_path << std::endl;
  }

  // Load YOLO model
  YoloModel model("~/code/quant_comp_v5m");

  // Load images
  std::vector<Image> images = YoloModel::load_images(images_path);

  // Run images
  std::vector<ImageResult> img_results = model.run_images(images);

  // Process results
  model.process_results(img_results, true, true);

  return EXIT_SUCCESS;
}