
<div align="center">
<h1 align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/5/59/Xilinx.svg" width="100" />
<br>
kr260-yolov5 
</h1>
<h3 align="center">🚀 Developed with the software and tools below.</h3>
<p align="center">

<img src="https://img.shields.io/badge/CMake-064F8C.svg?style=for-the-badge&logo=CMake&logoColor=white" alt="pipreqs" />
<img src="https://img.shields.io/badge/Git-F05032.svg?style=for-the-badge&logo=Git&logoColor=white" alt="pytorch2caffe" />
<img src="https://img.shields.io/badge/Markdown-000000.svg?style=for-the-badge&logo=Markdown&logoColor=white" alt="tornado" />
</p>

</div>

---
## 📚 Table of Contents
- [📚 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [⚙️ Project Structure](#-project-structure)
- [🧩 Modules](#-modules)
- [🚀 Getting Started](#-getting-started)
  - [✅ Prerequisites](#-prerequisites)
  - [💻 Installation](#-installation)
  - [🤖 Run demo server on KR260 board](#-run-demo-server-on-kr260-board)
  - [🤖 Run demo OBC on Host (Unix based OS)](#-run-demo-obc-on-host-unix-based-os)
  - [🧪 Running Benchmark on KR260 board](#-running-benchmark-on-kr260-board)

---

## 📍 Overview

This repository contains the development efforts for running a YOLOv5 model on a KR260 board.

---

<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-github-open.svg" width="80" />

## ⚙️ Project Structure

```bash
.
├── benchmark.cpp
├── board.cpp
├── CMakeLists.txt
├── host.cpp
├── message.proto
├── quant_comp_v5m
│   ├── quant_comp_v5m.classcsv
│   ├── quant_comp_v5m_dylan.xmodel
│   ├── quant_comp_v5m_old.prototxt
│   ├── quant_comp_v5m_old.xmodel
│   ├── quant_comp_v5m.prototxt
│   ├── quant_comp_v5m.xmodel
│   └── quant_comp_v5m_xview_qat.xmodel
├── scenes
│   ├── lb_1.png
│   ├── lb_2.png
│   ├── lb_3.png
│   ├── lb_4.png
│   ├── sfbay_1.png
│   ├── sfbay_2.png
│   ├── sfbay_3.png
│   └── sfbay_4.png
├── Server.hpp
├── YoloModel.cpp
└── YoloModel.hpp

2 directories, 23 files
```
---

<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-src-open.svg" width="80" />

## 💻 Modules
<details closed><summary>.</summary>

| File          | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|:--------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Server.hpp    | This code is a server class that creates a TCP socket to listen for incoming connections, sets the SO_REUSEADDR option to allow reuse of the same address and port, binds the socket to a port, listens for incoming connections, accepts an incoming connection, reads a message from the socket, parses the message from the byte array, serializes the message to a byte array, sends the message size to the socket, and sends the message data to the socket. |
| board.cpp     | This code is a server program that uses a YOLO model to detect objects in images from a camera. It includes functions to generate random numbers, load images, package images, build a reply, and start a server.                                                                                                                                                                                                                                                  |
| YoloModel.hpp | This code is a class for running YOLOv3 object detection on images. It includes functions for loading images, running the model, and processing the results.                                                                                                                                                                                                                                                                                                       |
| host.cpp      | This code creates a TCP socket to connect to a remote device, sends a request message, waits for a reply, and processes the reply.                                                                                                                                                                                                                                                                                                                                 |
| YoloModel.cpp | This code is for a YoloModel class which is used to load images, run the YOLO model on them, and process the results. It includes functions to check if a path is a file or directory, get absolute paths, check if a file is an image, get classes from a csv file, draw bounding boxes, and save images.                                                                                                                                                         |
| message.proto | This code defines a message called MyMessage which contains an enum CommandType, two messages Request and Reply, and several fields such as id, time_sent, command, request, and reply.                                                                                                                                                                                                                                                                            |
| .clang-format | This code is a style guide for writing code in the Google style. It provides guidelines for formatting, naming conventions, and other coding conventions to ensure code is written in a consistent and readable manner.                                                                                                                                                                                                                                            |
| benchmark.cpp | This code loads a YOLO model from a specified path, loads images from a specified path, runs the images through the model, and processes the results.                                                                                                                                                                                                                                                                                                              |

</details>

<details closed><summary>Quant_comp_v5m</summary>

| File                        | Summary                                                                                                                                                                                                  |
|:----------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| quant_comp_v5m_old.prototxt | This code is a model for YOLOv3, a type of object detection algorithm. It has 80 classes, 3 anchors, and a confidence threshold of 0. 1 and a non-maximum suppression threshold of 0. 1.                 |
| quant_comp_v5m.classcsv     | This code simulates a ship docking at a dock. It allows the user to control the ship's speed and direction as it approaches the dock, and provides feedback on the ship's progress.                      |
| quant_comp_v5m.prototxt     | This code is a model for YOLOv3, a type of object detection algorithm. It has two classes and three anchor boxes, with a confidence threshold of 0. 25 and a non-maximum suppression threshold of 0. 45. |

</details>
<hr />

## 🚀 Getting Started

### ✅ Prerequisites

Before you begin, ensure that you have the following prerequisites installed:
> `cmake`
> `cpp`
> `vitis-ai-library`
> `opencv`

### 💻 Installation

1. Clone the readme-ai repository:
```sh
git clone https://github.com/shamoryj/kr260-yolov5
```

2. Change to the project directory:
```sh
cd kr260-yolov5
```

3. Compile the programs:
```sh
mkdir build && cd build
cmake -S .. -B .
```

### 🤖 Run demo server on KR260 board

```sh
./board
```

### 🤖 Run demo OBC on Host (Unix based OS)

```sh
./host
```

### 🧪 Running Benchmark on KR260 board
```sh
./benchmark
```

<hr />
