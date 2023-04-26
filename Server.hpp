#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <ctime>
#include <iostream>

#include "message.pb.h"

double secondsSinceEpoch() {
  // get the current time
  std::time_t current_time = std::time(nullptr);

  // convert to seconds since the epoch
  return std::difftime(current_time, 0);
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
