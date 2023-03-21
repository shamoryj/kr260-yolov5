#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <thread>
#include <vector>

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

double secondsSinceEpoch() {
  // get the current time
  std::time_t current_time = std::time(nullptr);

  // convert to seconds since the epoch
  return std::difftime(current_time, 0);
}

bool sendMessage(MyMessage &message, int sockfd) {
  // Serialize the message to a byte array
  message.set_time_sent(secondsSinceEpoch());
  size_t size = message.ByteSizeLong();
  char messageData[size];
  message.SerializeToArray(messageData, size);

  // Send the message to the socket
  ssize_t numSent = send(sockfd, messageData, size, 0);
  if (numSent == -1) {
    std::cerr << "Error: Failed to send message" << std::endl;
    return false;
  }
  return true;
}

bool receiveMessage(MyMessage &message, int sockfd) {
  // Read size of data from the socket
  size_t size;
  ssize_t numRecv = recv(sockfd, &size, sizeof(size), 0);
  if (numRecv == -1) {
    std::cerr << "Error: Failed to receive size of message" << std::endl;
    return false;
  }

  // Read the message data from the socket
  char *buffer = (char *)malloc(size);
  size_t bytesReceived = 0;
  while (bytesReceived < size) {
    numRecv = recv(sockfd, buffer + bytesReceived, size - bytesReceived, 0);
    if (numRecv == -1) {
      std::cerr << "Error: Failed to receive message" << std::endl;
      free(buffer);
      return false;
    }
    bytesReceived += numRecv;
  }

  // Parse the message from the received data
  if (!message.ParseFromArray(buffer, size)) {
    free(buffer);
    std::cerr << "Error: Failed to parse message" << std::endl;
    return false;
  }
  free(buffer);
  return true;
}

// Function to save a MyImage message to a file
void save_image(const std::string &filename, const MyMessage_Image &image) {
  // Convert the image data to a cv::Mat
  cv::Mat img(image.height(), image.width(), CV_8UC(image.channels()));
  std::memcpy(img.data, image.data().data(), image.data().size());

  // Encode the image to a JPEG file
  std::vector<uint8_t> buffer;
  cv::imencode(".jpg", img, buffer);
  std::ofstream file(filename, std::ios::binary);
  file.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
}

int main(int argc, char *argv[]) {
  std::string board_ip = "10.0.40.40";
  short board_port = 12345;

  // Create a TCP socket to connect to the remote device
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == -1) {
    std::cerr << "Error: Failed to create socket" << std::endl;
    return 1;
  }

  // Connect to the remote device
  struct sockaddr_in remoteAddr {};
  memset(&remoteAddr, 0, sizeof(remoteAddr));
  remoteAddr.sin_family = AF_INET;
  remoteAddr.sin_addr.s_addr = inet_addr(board_ip.c_str());
  remoteAddr.sin_port = htons(board_port);
  if (connect(sockfd, (struct sockaddr *)&remoteAddr, sizeof(remoteAddr)) ==
      -1) {
    std::cerr << "Error: Failed to connect to remote device" << std::endl;
    return 1;
  }

  RandomGenerator rng;
  for (int id = 0;; id++) {
    // Create a message to send to the board
    MyMessage request;
    request.set_command(MyMessage::REQUEST);

    request.set_id(id);
    request.mutable_request()->set_get_image(true);
    request.mutable_request()->set_get_bounding_box_image(true);

    // Send the request to the board
    if (!sendMessage(request, sockfd)) {
      break;
    }
    std::cout << "Sent request: " << request.id() << std::endl;

    // Wait for a reply from the board
    MyMessage reply;
    if (!receiveMessage(reply, sockfd)) {
      break;
    }
    std::cout << "Received reply: " << reply.id() << std::endl;
    // Process the reply
    if (reply.command() == MyMessage::REPLY) {
      std::cout << "Time sent: " << reply.time_sent() << std::endl;
      for (const auto &box : reply.reply().bounding_boxes()) {
        std::cout << "label: " << box.label() << ", x_min: " << box.x_min()
                  << ", y_min: " << box.y_min() << ", x_max: " << box.x_max()
                  << ", y_max: " << box.y_max()
                  << ", confidence: " << box.confidence() << std::endl;
      }
      if (request.request().get_image()) {
        if (reply.reply().has_image()) {
          save_image(std::to_string(reply.id()) + ".jpg",
                     reply.reply().image());
        } else {
          std::cerr << "Error: Missing requested image" << std::endl;
        }
      }
      if (request.request().get_bounding_box_image()) {
        if (reply.reply().has_bounding_box_image()) {
          save_image(std::to_string(reply.id()) + "_bbox.jpg",
                     reply.reply().bounding_box_image());
        } else {
          std::cerr << "Error: Missing requested bounding box image"
                    << std::endl;
        }
      }
    } else {
      std::cerr << "Error: Unsupported command" << std::endl;
    }

    // Sleep for 5 to 20 seconds
    unsigned int seconds = rng.next_in_range(5, 20);
    std::cout << "Sleeping for " << seconds << " seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    std::cout << "Done sleeping." << std::endl;
  }

  // Close the socket
  close(sockfd);

  return 0;
}
