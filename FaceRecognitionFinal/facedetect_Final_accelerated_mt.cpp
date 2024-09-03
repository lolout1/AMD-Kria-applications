#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include "xf_resize.hpp"
#include "xf_boundingbox.hpp"
#include <vitis/ai/facedetectrecog.hpp>
#include <vitis/ai/profiling.hpp>
#include <opencv2/opencv.hpp>

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

// ThreadPool class to manage a pool of threads
class ThreadPool {
public:
    ThreadPool(size_t threads);
    template<class F> auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type>;
    ~ThreadPool();
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back([this] {
            for(;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                    if(this->stop && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
}

template<class F>
auto ThreadPool::enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
    using return_type = typename std::result_of<F()>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

float feature_norm(const int8_t *feature) {
  int sum = 0;
  for (int i = 0; i < 512; ++i) {
    sum += feature[i] * feature[i];
  }
  return 1.f / sqrt(sum);
}

/// This function is used for computing the dot product of two vectors
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

void process_image(int index, const cv::Mat &img_normal, 
                   FaceDetectRecog *detectrecog) {
    xf::cv::Mat<TYPE, SRC_ROWS, SRC_COLS, NPC> src_mat(img_normal.rows, img_normal.cols);
    xf::cv::Mat<TYPE, DST_ROWS, DST_COLS, NPC> dst_mat(DST_ROWS, DST_COLS);
    
    src_mat.copyTo(img_normal.data);
    xf::cv::resize<XF_INTERPOLATION_BILINEAR, TYPE, SRC_ROWS, SRC_COLS, DST_ROWS, DST_COLS, NPC>(src_mat, dst_mat);

    std::vector<xf::cv::Mat<TYPE, DST_ROWS, DST_COLS, NPC>> resized_images = {dst_mat};
    auto result_batch = detectrecog->run_fixed(resized_images);

    auto& image_resized = resized_images[0];
    std::cout << "Processing image: " << index << std::endl;

    for (size_t i = 0; i < result_batch[0].rects.size(); ++i) {
      xf::cv::Rect_<int> bbox(
        static_cast<int>(result_batch[0].rects[i].x * image_resized.cols),
        static_cast<int>(result_batch[0].rects[i].y * image_resized.rows),
        static_cast<int>(result_batch[0].rects[i].width * image_resized.cols),
        static_cast<int>(result_batch[0].rects[i].height * image_resized.rows)
      );

      xf::cv::Scalar<4, unsigned char> color(255, 0, 0, 0);
      xf::cv::boundingbox<TYPE, SRC_ROWS, SRC_COLS>(image_resized, &bbox, &color, 1);
    }

    for (size_t i = 0; i < result_batch[0].features.size(); ++i) {
      write_bin(&result_batch[0].features[i][0], 512, 
                ("feature-" + std::to_string(index) + "-" + std::to_string(i) + ".bin").c_str());
    }

    cv::Mat output_image(SRC_ROWS, SRC_COLS, CV_8UC3, image_resized.data);
    cv::imwrite("recog_result-" + std::to_string(index) + ".jpg", output_image);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image1> <image2> ..." << std::endl;
    exit(0);
  }

  // Create face detection and recognition model
  auto detectrecog = FaceDetectRecog::create("densebox_640_360", "face_landmark", "facerec_resnet20", true);

  ThreadPool pool(6);  // Create a thread pool with 6 threads

  // Process each image in parallel
  std::vector<std::future<void>> futures;
  for (int i = 1; i < argc; ++i) {
    cv::Mat img_normal = cv::imread(argv[i]);
    if (img_normal.empty()) {
      std::cerr << "Error: Could not read image " << argv[i] << std::endl;
      continue;
    }

    futures.push_back(pool.enqueue([=] { 
      process_image(i, img_normal, detectrecog.get());
    }));
  }

  // Wait for all tasks to complete
  for (auto &f : futures) {
    f.get();
  }

  return 0;
}
