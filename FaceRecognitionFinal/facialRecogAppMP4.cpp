#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vitis/ai/facedetectrecog.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <chrono>
#include <sstream>
#include <thread>

// Function to compute the norm of a feature vector
float feature_norm(const int8_t *feature) {
    int sum = 0;
    for (int i = 0; i < 512; ++i) {
        sum += feature[i] * feature[i];
    }
    return 1.f / sqrt(sum);
}

// Function to compute the dot product of two vectors
static float feature_dot(const int8_t *f1, const int8_t *f2) {
    int dot = 0;
    for (int i = 0; i < 512; ++i) {
        dot += f1[i] * f2[i];
    }
    return static_cast<float>(dot);
}

// Function to calculate similarity score
float feature_compare(const int8_t *feature, const int8_t *feature_lib) {
    float norm = feature_norm(feature);
    float feature_norm_lib = feature_norm(feature_lib);
    return feature_dot(feature, feature_lib) * norm * feature_norm_lib;
}

// Function to map the similarity score
float score_map(float score) {
    return 1.0 / (1 + exp(-12.4 * score + 3.763));
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
            float similarity_score = feature_compare(result.features[i].data(), ref.second.data());
            if (similarity_score > max_similarity) {
                max_similarity = similarity_score;
                best_match = ref.first;
            }
        }

        float mapped_score = score_map(max_similarity);
        std::string similarity_text = (mapped_score > 0.65 ? "Abheek" : "Unknown") + std::string(": ") + std::to_string(mapped_score);

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

// Parse resolution from a string formatted as "WIDTHxHEIGHT"
bool parse_resolution(const std::string& res_str, int& width, int& height) {
    std::stringstream ss(res_str);
    char x;
    return (ss >> width >> x >> height) && (x == 'x');
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    // Hardcoded model paths
    std::string detection_model = "densebox_320_320";
    std::string landmark_model = "face_landmark";
    std::string recognition_model = "facerec-resnet20_mixed_pt";

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <video_source> <reference_image1> [<reference_image2> ...] [-f WIDTHxHEIGHT] [output_file]" << std::endl;
        return 1;
    }

    std::string video_source = argv[1];
    int output_width = 1920;  // Default width
    int output_height = 1080; // Default height

    std::string output_file = "output.mp4";  // Default output file

    // Parse command-line arguments for reference images and optional output size
    std::vector<std::pair<std::string, std::array<int8_t, 512>>> reference_features;
    auto face_detect_recog = vitis::ai::FaceDetectRecog::create(detection_model, landmark_model, recognition_model, true);

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            if (!parse_resolution(argv[++i], output_width, output_height)) {
                std::cerr << "Invalid resolution format. Use WIDTHxHEIGHT." << std::endl;
                return 1;
            }
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else {
            cv::Mat reference_image = cv::imread(arg);
            if (reference_image.empty()) {
                std::cerr << "Error: Could not load reference image " << arg << std::endl;
                continue;
            }
            auto ref_results = face_detect_recog->run_fixed(reference_image);
            if (!ref_results.features.empty()) {
                reference_features.emplace_back(arg, ref_results.features[0]);
            } else {
                std::cerr << "Error: Could not extract features from reference image " << arg << std::endl;
            }
        }
    }

    // Open video capture
    cv::VideoCapture cap(video_source);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source " << video_source << std::endl;
        return 1;
    }

    // Setup video writer
    cv::VideoWriter writer(output_file, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 30, cv::Size(output_width, output_height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open video writer with file name " << output_file << std::endl;
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
        batch_frames[frame_count % batch_size] = frame;
        frame_count++;

        if (frame_count % batch_size == 0) {
            auto t2 = std::chrono::high_resolution_clock::now();
            auto results_batch = face_detect_recog->run_fixed(batch_frames);
            auto t3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dpu_time = t3 - t2;
            std::cout << "DPU Processing Time: " << dpu_time.count() << " seconds" << std::endl;

            // Process results and display
            for (size_t i = 0; i < batch_size; ++i) {
                auto display_frame = process_result(batch_frames[i], results_batch[i], reference_features, fps);

                // Resize to specified output resolution
                cv::resize(display_frame, display_frame, cv::Size(output_width, output_height));

                writer.write(display_frame);  // Write the processed frame to the video file

                cv::imshow("Face Detection and Recognition", display_frame);
                if (cv::waitKey(1) >= 0) break;  // Exit on any key press
            }

            auto t4 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> loop_time = t4 - t1;
            std::cout << "Total Loop Time: " << loop_time.count() << " seconds" << std::endl;

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
    writer.release();  // Release the video writer
    cv::destroyAllWindows();
    return 0;
}
//./face_recognition_app /dev/video0 reference1.jpg reference2.jpg -f 1920x1080 -o output.mp4
