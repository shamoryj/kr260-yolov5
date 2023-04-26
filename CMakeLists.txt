cmake_minimum_required(VERSION 3.5)

# Set project name and C++ version
project(k26_yolov5)
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Find the OpenCV library
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find the Threads library
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Find the Protocol Buffers library
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

# Add the generated protobuf message files to the project
file(GLOB PROTO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# Add the board executable
add_executable(board board.cpp Server.hpp YoloModel.cpp ${PROTO_SRCS} ${PROTO_HDRS})
# Link against the Vitis AI libraries
target_link_libraries(board vitis_ai_library-yolov3)
target_link_libraries(board vitis_ai_library-dpu_task)
target_link_libraries(board vitis_ai_library-xnnpp)
target_link_libraries(board vitis_ai_library-model_config)
target_link_libraries(board vitis_ai_library-math)
# Link against other Xilinx libraries
target_link_libraries(board vart-util)
target_link_libraries(board xir)
# Link against Threads library
target_link_libraries(board Threads::Threads)
target_link_libraries(board json-c)
target_link_libraries(board glog)
# Link against OpenCV libraries
target_link_libraries(board ${OpenCV_LIBS})
target_link_libraries(board opencv_core)
target_link_libraries(board opencv_videoio)
target_link_libraries(board opencv_imgproc)
target_link_libraries(board opencv_imgcodecs)
target_link_libraries(board opencv_highgui)
# Link against the Protocol Buffers library
target_link_libraries(board ${PROTOBUF_LIBRARIES})

# Add the host executable
add_executable(host host.cpp YoloModel.hpp ${PROTO_SRCS} ${PROTO_HDRS})
# Link against OpenCV libraries
target_link_libraries(host ${OpenCV_LIBS})
target_link_libraries(host opencv_core)
target_link_libraries(host opencv_videoio)
target_link_libraries(host opencv_imgproc)
target_link_libraries(host opencv_imgcodecs)
target_link_libraries(host opencv_highgui)
# Link against the Protocol Buffers library
target_link_libraries(host ${PROTOBUF_LIBRARIES})

# Add the benchmark executable
add_executable(benchmark benchmark.cpp YoloModel.cpp)
# Link against the Vitis AI libraries
target_link_libraries(benchmark vitis_ai_library-yolov3)
target_link_libraries(benchmark vitis_ai_library-dpu_task)
target_link_libraries(benchmark vitis_ai_library-xnnpp)
target_link_libraries(benchmark vitis_ai_library-model_config)
target_link_libraries(benchmark vitis_ai_library-math)
# Link against other Xilinx libraries
target_link_libraries(benchmark vart-util)
target_link_libraries(benchmark xir)
# Link against Threads library
target_link_libraries(benchmark Threads::Threads)
target_link_libraries(benchmark json-c)
target_link_libraries(benchmark glog)
# Link against OpenCV libraries
target_link_libraries(benchmark ${OpenCV_LIBS})
target_link_libraries(benchmark opencv_core)
target_link_libraries(benchmark opencv_videoio)
target_link_libraries(benchmark opencv_imgproc)
target_link_libraries(benchmark opencv_imgcodecs)
target_link_libraries(benchmark opencv_highgui)
