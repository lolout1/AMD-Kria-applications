/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <fstream>
#include <string>

#include "xf_resize.hpp"
#include "xf_boundingbox.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <stdio.h>
#include <unistd.h>

#include <vitis/ai/facedetectrecog.hpp>
#include <vitis/ai/profiling.hpp>

using namespace vitis::ai;
using namespace std;
using namespace xf::cv;

// Constants and Types
constexpr int TYPE = XF_8UC3;
constexpr int SRC_ROWS = 1080;
constexpr int SRC_COLS = 1920;
constexpr int DST_ROWS = 320;
constexpr int DST_COLS = 320;
constexpr int NPC = XF_NPPC1;

float feature_norm(const int8_t *feature) {
  int sum = 0;
  for (int i = 0; i < 512; ++i) {
    sum += feature[i] * feature[i];
  }
  return 1.f / sqrt(sum);
}

/// This function is used for computing dot product of two vectors
static float feature_dot(const int8_t *f1, const int8_t *f2) {
  int dot = 0;
  for (int i = 0; i < 512; ++i) {
    dot += f1[i] * f2[i];
  }
  return static_cast<float>(dot);
}

float feature_compare(const int8_t *feature, const int8_t *feature_lib) {
  float norm = feature_norm(feature);
  float feature_norm_lib = feature_norm(feature_lib);
  return feature_dot(feature, feature_lib) * norm * feature_norm_lib;
}

float score_map(float score) { 
  return 1.0 / (1 + exp(-12.4 * score + 3.763)); 
}

void read_bin(int8_t* dst, int size, const char* file_path) {
  std::ifstream in(file_path, ios::in | ios::binary);
  if (!in.is_open()) {
    std::cerr << "Error: Could not open file " << file_path << std::endl;
    exit(0);
  } else {
    for (int i = 0; i < size; i++) {
      in.read(reinterpret_cast<char*>(dst + i), sizeof(int8_t));
    }
  }
}

void write_bin(const int8_t *src, int size, const char * file_path) {
  std::cout << "Output path: " << file_path << std::endl;
  std::ofstream out(file_path, ios::out|ios::binary);
  out.write(reinterpret_cast<const char*>(src), sizeof(int8_t) * size);
  out.close();
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image1> <image2> ..." << std::endl;
    exit(0);
  }

  std::vector<xf::cv::Mat<TYPE, SRC_ROWS, SRC_COLS, NPC>> images;
  std::vector<xf::cv::Mat<TYPE, DST_ROWS, DST_COLS, NPC>> resized_images;

  // Load and resize images
  for (int i = 1; i < argc; ++i) {
    cv::Mat img_normal = cv::imread(argv[i]);
    if (img_normal.empty()) {
      std::cerr << "Error: Could not read image " << argv[i] << std::endl;
      exit(2);
    }
    xf::cv::Mat<TYPE, SRC_ROWS, SRC_COLS, NPC> src_mat(img_normal.rows, img_normal.cols);
    xf::cv::Mat<TYPE, DST_ROWS, DST_COLS, NPC> dst_mat(DST_ROWS, DST_COLS);

    src_mat.copyTo(img_normal.data);
    xf::cv::resize<XF_INTERPOLATION_BILINEAR, TYPE, SRC_ROWS, SRC_COLS, DST_ROWS, DST_COLS, NPC>(src_mat, dst_mat);

    images.push_back(src_mat);
    resized_images.push_back(dst_mat);
  }

  // Create face detection and recognition model
  auto detectrecog = FaceDetectRecog::create("densebox_640_360", "face_landmark", "facerec_resnet20", true);

  // Perform face detection and recognition
  __TIC__(RECOG_MAT_FIXED_NORMAL)
  auto result_batch = detectrecog->run_fixed(resized_images);
  __TOC__(RECOG_MAT_FIXED_NORMAL)

  // Process results
  for (size_t n = 0; n < result_batch.size(); ++n) {
    auto& image_resized = resized_images[n];
    std::cout << "Processing image: " << n << std::endl;

    for (size_t i = 0; i < result_batch[n].rects.size(); ++i) {
      xf::cv::Rect_<int> bbox(
        static_cast<int>(result_batch[n].rects[i].x * image_resized.cols),
        static_cast<int>(result_batch[n].rects[i].y * image_resized.rows),
        static_cast<int>(result_batch[n].rects[i].width * image_resized.cols),
        static_cast<int>(result_batch[n].rects[i].height * image_resized.rows)
      );

      xf::cv::Scalar<4, unsigned char> color(255, 0, 0, 0);
      xf::cv::boundingbox<TYPE, SRC_ROWS, SRC_COLS>(image_resized, &bbox, &color, 1);
    }

    for (size_t i = 0; i < result_batch[n].features.size(); ++i) {
      write_bin(&result_batch[n].features[i][0], 512, ("feature-" + std::to_string(n) + "-" + std::to_string(i) + ".bin").c_str());
    }

    cv::Mat output_image(SRC_ROWS, SRC_COLS, CV_8UC3, image_resized.data);
    cv::imwrite("recog_result-" + std::to_string(n) + ".jpg", output_image);
  }

  return 0;
}
