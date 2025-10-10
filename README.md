# AMD Kria Applications

**[Youtube video link](https://youtu.be/-e7vtFMeb6o)** | **[Benchmarks](https://github.com/lolout1/AMD-Kria-applications/tree/main/bench)**

This repository showcases high-performance embedded vision applications built on the AMD Kria KV260 Vision AI Starter Kit. The primary project is a custom hardware-accelerated facial recognition system that processes USB webcam or MP4 video input in real-time with near-zero latency.

---

## Features

- **Real-time facial recognition** with user-configurable reference images
- **Hardware-accelerated pipeline** using DPU (Deep Learning Processing Unit) on FPGA fabric
- **Zero-copy DMA architecture** eliminating CPU memory bottlenecks
- **Multiple output options**: HDMI, DisplayPort, or X11 forwarding
- **Near-native FPS performance**: Matches webcam capabilities (30-60 FPS) with sub-2ms latency

### Zero-Copy Pipeline Architecture

Custom Vivado platform with hardware preprocessing pipeline eliminates all CPU memory copies. Camera frames flow through Video Processing Subsystem (hardware resize/color conversion) → Vitis HLS preprocessing kernels (normalization) → AXI DMA → device-only DDR buffers shared with DPU. The ARM CPU only touches final inference results (bounding boxes and embeddings), never the raw pixel data. **Result: 60% memory bandwidth reduction and 2ms end-to-end latency.**

---

## Quick Start

### Prerequisites
- Kria KV260 with PetaLinux 2022.2+
- USB webcam or MP4 video file
- Reference images of faces to recognize
- HDMI/DP display or X11 forwarding setup

### Usage
```bash
# Basic usage with webcam
./facialdetect_recog /dev/video2 reference_face.jpg -f 1920x1080

# General syntax
./facialdetect_recog <video_source> <reference_image.jpg> -f <resolution>

# Examples
./facialdetect_recog /dev/video0 person1.jpg person2.jpg -f 1920x1080
./facialdetect_recog input_video.mp4 john.jpg jane.jpg -f 1280x720
