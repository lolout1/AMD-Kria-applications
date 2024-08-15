#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <future>
#include <glog/logging.h>
#include <vitis/ai/facedetectrecog.hpp>
#include <mutex>
#include <condition_variable>

// Declare and define the ThreadPool class (Ensure this is before its usage)
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

// ThreadPool implementation
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

// Function to compute the norm of a feature vector
float compute_feature_norm(const int8_t *feature) {
    int64_t sum = 0;
    for (int i = 0; i < 512; ++i) {
        sum += static_cast<int64_t>(feature[i]) * feature[i];
    }
    return 1.f / sqrt(static_cast<float>(sum));
}

// Function to compute the dot product of two vectors
float compute_feature_dot(const int8_t *f1, const int8_t *f2) {
    int64_t dot = 0;
    for (int i = 0; i < 512; ++i) {
        dot += static_cast<int64_t>(f1[i]) * f2[i];
    }
    return static_cast<float>(dot);
}

// Function to calculate similarity score
float compute_similarity_score(const int8_t *feature, const int8_t *reference_feature) {
    float norm = compute_feature_norm(feature);
    float reference_norm = compute_feature_norm(reference_feature);
    return compute_feature_dot(feature, reference_feature) * norm * reference_norm;
}

// Function to map the similarity score to a [0, 1] range
float map_similarity_score(float score) {
    return 1.0f / (1 + exp(-12.4f * score + 3.763f));
}

// Function to process a single face result
void process_face(const vitis::ai::FaceDetectRecogFixedResult &result,
                  const std::vector<std::pair<std::string, std::array<int8_t, 512>>>& reference_features,
                  cv::Mat &image, int index) {

    const auto &rect = result.rects[index];

    // Find the reference image with the highest similarity score
    float max_similarity = -1.0f;
    std::string best_match = "Unknown";
    for (const auto& ref : reference_features) {
        float similarity_score = compute_similarity_score(result.features[index].data(), ref.second.data());
        if (similarity_score > max_similarity) {
            max_similarity = similarity_score;
            best_match = ref.first;
        }
    }

    float mapped_score = map_similarity_score(max_similarity);
    std::string similarity_text = (mapped_score > 0.5f ? "Abheek" : "Unknown") + std::string(": ") + std::to_string(mapped_score);

    // Draw a rectangle around each detected face with the best match similarity score
    cv::rectangle(image,
                  cv::Rect(cv::Point(rect.x * image.cols, rect.y * image.rows),
                           cv::Size(static_cast<int>(rect.width * image.cols),
                                    static_cast<int>(rect.height * image.rows))),
                  cv::Scalar(255, 0, 0), 2);  // Rectangle color (blue)

    // Display the best match and similarity score on the image inside the bounding box
    cv::putText(image, similarity_text,
                cv::Point(rect.x * image.cols, rect.y * image.rows - 5),  // Position the text above the box
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);  // Text color (green)
}

// Function to process results and display
cv::Mat process_results(cv::Mat &frame, const vitis::ai::FaceDetectRecogFixedResult &result,
                       const std::vector<std::pair<std::string, std::array<int8_t, 512>>>& reference_features,
                       ThreadPool &pool) {

    cv::Mat output_frame = frame.clone();
    std::vector<std::future<void>> futures;

    // Iterate over detected faces and process each in parallel
    for (size_t i = 0; i < result.rects.size(); ++i) {
        futures.push_back(pool.enqueue([&, i] {
            process_face(result, reference_features, output_frame, i);
        }));
    }

    // Wait for all asynchronous tasks to complete
    for (auto &f : futures) {
        f.get();
    }

    return output_frame;  // Return the processed image
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_source> [<reference_image1> [<reference_image2> ...]]" << std::endl;
        std::cerr << "<video_source> should be the path to the input video file." << std::endl;
        return 1;
    }

    std::string video_source = argv[1];

    // Parse command-line arguments for reference images
    std::vector<std::pair<std::string, std::array<int8_t, 512>>> reference_features;
    auto face_detect_recog = vitis::ai::FaceDetectRecog::create("densebox_320_320", "face_landmark", "facerec-resnet20_mixed_pt", true);

    // Print input dimensions required by the model
    int input_width = face_detect_recog->getInputWidth();
    int input_height = face_detect_recog->getInputHeight();
    std::cout << "Model Input Size: Width = " << input_width << ", Height = " << input_height << std::endl;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        cv::Mat reference_image = cv::imread(arg);
        if (reference_image.empty()) {
            std::cerr << "Error: Could not load reference image " << arg << std::endl;
            continue;
        }
        auto ref_results = face_detect_recog->run_fixed(reference_image);
        if (!ref_results.features.empty()) {
            reference_features.push_back({arg, ref_results.features[0]});
        }
    }

    cv::VideoCapture cap(video_source);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source " << video_source << std::endl;
        return 1;
    }

    ThreadPool pool(8);  // Initialize the thread pool with 8 threads
    cv::Mat frame;
    auto start_time = std::chrono::steady_clock::now();
    int frame_count = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Resize the frame to the model's input dimensions
        cv::resize(frame, frame, cv::Size(input_width, input_height));

        auto frame_start_time = std::chrono::steady_clock::now();
        auto results = face_detect_recog->run_fixed(frame);
        auto processed_frame = process_results(frame, results, reference_features, pool);

        // Display the processed frame
        cv::imshow("Face Detection and Recognition", processed_frame);

        // Calculate and display FPS
        frame_count++;
        auto frame_end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> frame_duration = frame_end_time - frame_start_time;
        double fps = 1.0 / frame_duration.count();
        std::string fps_text = "FPS: " + std::to_string(fps);
        cv::putText(processed_frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);  // FPS text color (yellow)

        // Display the frame with FPS
        cv::imshow("Face Detection and Recognition", processed_frame);

        // Exit if the user presses the 'q' key
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    double average_fps = frame_count / elapsed_time;
    std::cout << "Average FPS: " << average_fps << std::endl;

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
