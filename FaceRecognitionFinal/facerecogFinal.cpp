#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vitis/ai/facedetectrecog.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <chrono>

// Function to normalize feature vector
std::vector<float> normalize_feature(const std::array<int8_t, 512>& feature) {
    std::vector<float> normalized(feature.size());
    float norm = 0.0;
    for (size_t i = 0; i < feature.size(); ++i) {
        norm += feature[i] * feature[i];
    }
    norm = std::sqrt(norm);
    for (size_t i = 0; i < feature.size(); ++i) {
        normalized[i] = static_cast<float>(feature[i]) / norm;
    }
    return normalized;
}

// Function to calculate cosine similarity between two feature vectors
float calculate_cosine_similarity(const std::array<int8_t, 512>& f1, const std::array<int8_t, 512>& f2) {
    auto nf1 = normalize_feature(f1);
    auto nf2 = normalize_feature(f2);

    float dot_product = 0.0;
    for (size_t i = 0; i < nf1.size(); ++i) {
        dot_product += nf1[i] * nf2[i];
    }
    return dot_product;
}

// Function to process face detection and recognition results
cv::Mat process_result(cv::Mat &m1, const vitis::ai::FaceDetectRecogFixedResult &result, const std::vector<std::pair<std::string, std::array<int8_t, 512>>>& reference_features, double fps) {
    cv::Mat image = m1.clone();

    // Iterate over detected faces and draw bounding boxes with similarity scores
    for (size_t i = 0; i < result.rects.size(); ++i) {
        const auto &r = result.rects[i];

        // Find the reference image with the highest similarity score
        float max_similarity = -1.0;
        std::string best_match = "Unknown";
        for (const auto& ref : reference_features) {
            float similarity_score = calculate_cosine_similarity(result.features[i], ref.second);
            if (similarity_score > max_similarity) {
                max_similarity = similarity_score;
                best_match = ref.first;
            }
        }

        std::string similarity_text = (max_similarity > 0.65 ? std::string("Abheek") : std::string("Unknown")) + ": " + std::to_string(max_similarity);

        // Draw a rectangle around each detected face with the best match similarity score
        cv::rectangle(image,
                      cv::Rect(cv::Point(r.x * image.cols, r.y * image.rows),
                               cv::Size(static_cast<int>(r.width * image.cols),
                                        static_cast<int>(r.height * image.rows))),
                      cv::Scalar(255, 0, 0), 2);  // Rectangle color (blue)

        // Display the best match and similarity score on the image inside the bounding box
        cv::putText(image, similarity_text,
                    cv::Point(r.x * image.cols, r.y * image.rows - 5),  // Position the text above the box
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);  // Text color (green)
    }

    // Display the FPS on the top-left corner of the image
    std::string fps_text = "FPS: " + std::to_string(fps);
    cv::putText(image, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);  // FPS text color (yellow)

    return image;  // Return the processed image
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <detection_model> <landmark_model> <recognition_model> <video_source> <reference_image1> [<reference_image2> ...]" << std::endl;
        return 1;
    }

    std::string detection_model = argv[1];
    std::string landmark_model = argv[2];
    std::string recognition_model = argv[3];
    std::string video_source = argv[4];

    // Load and extract features from reference images
    std::vector<std::pair<std::string, std::array<int8_t, 512>>> reference_features;
    auto face_detect_recog = vitis::ai::FaceDetectRecog::create(detection_model, landmark_model, recognition_model, true);

    for (int i = 5; i < argc; ++i) {
        cv::Mat reference_image = cv::imread(argv[i]);
        if (reference_image.empty()) {
            std::cerr << "Error: Could not load reference image " << argv[i] << std::endl;
            continue;
        }
        auto ref_results = face_detect_recog->run_fixed(reference_image);
        if (!ref_results.features.empty()) {
            reference_features.emplace_back(argv[i], ref_results.features[0]);
        } else {
            std::cerr << "Error: Could not extract features from reference image " << argv[i] << std::endl;
        }
    }

    // Open video capture
    cv::VideoCapture cap(video_source);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source " << video_source << std::endl;
        return 1;
    }

    // Get batch size from the model
    size_t batch_size = face_detect_recog->get_input_batch();
    std::cout << "Optimal Batch Size: " << batch_size << std::endl;

    std::vector<cv::Mat> batch_frames(batch_size);

    cv::Mat frame;
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    double fps = 0.0;

    while (cap.read(frame)) {
        auto t1 = std::chrono::high_resolution_clock::now();

        // Store captured frames into the batch
        batch_frames[frame_count % batch_size] = frame.clone();
        frame_count++;

        if (frame_count % batch_size == 0) {
            // Measure time for running detection and recognition in batch
            auto t2 = std::chrono::high_resolution_clock::now();
            auto results_batch = face_detect_recog->run_fixed(batch_frames);
            auto t3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dpu_time = t3 - t2;
            std::cout << "DPU Processing Time: " << dpu_time.count() << " seconds" << std::endl;

            // Process results and display
            for (size_t i = 0; i < batch_size; ++i) {
                auto display_frame = process_result(batch_frames[i], results_batch[i], reference_features, fps);
                cv::imshow("Face Detection and Recognition", display_frame);
                if (cv::waitKey(1) >= 0) break;  // Exit on any key press
            }

            // Measure total loop time including processing and display
            auto t4 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> loop_time = t4 - t1;
            std::cout << "Total Loop Time: " << loop_time.count() << " seconds" << std::endl;

            // Calculate and display FPS
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            if (elapsed.count() > 1.0) {
                fps = frame_count / elapsed.count();
                std::cout << "FPS: " << fps << std::endl;
                frame_count = 0;
                start_time = std::chrono::high_resolution_clock::now();
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
