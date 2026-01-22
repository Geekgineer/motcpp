#include <motcpp/trackers/bytetrack.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

int main() {
    std::cout << "motcpp - Simple Tracking Example\n";
    std::cout << "===============================\n\n";
    
    // Create tracker
    motcpp::trackers::ByteTrack tracker(
        0.3f,  // det_thresh
        30,    // max_age
        50,    // max_obs
        3,     // min_hits
        0.3f,  // iou_threshold
        false, // per_class
        80,    // nr_classes
        "iou", // asso_func
        false, // is_obb
        0.1f,  // min_conf
        0.5f,  // track_thresh
        0.8f,  // match_thresh
        30,    // track_buffer
        30     // frame_rate
    );
    
    // Create a test image (640x480)
    cv::Mat img = cv::Mat::zeros(480, 640, CV_8UC3);
    img.setTo(cv::Scalar(100, 100, 100));
    
    // Create synthetic detections: [x1, y1, x2, y2, conf, cls]
    // Simulate 3 objects moving across frames
    Eigen::MatrixXf detections(3, 6);
    
    std::cout << "Processing 10 frames with synthetic detections...\n\n";
    
    for (int frame = 0; frame < 10; ++frame) {
        // Object 1: moving right
        detections(0, 0) = 50.0f + frame * 10.0f;  // x1
        detections(0, 1) = 100.0f;                 // y1
        detections(0, 2) = 150.0f + frame * 10.0f; // x2
        detections(0, 3) = 200.0f;                 // y2
        detections(0, 4) = 0.8f;                   // conf
        detections(0, 5) = 0.0f;                   // cls
        
        // Object 2: moving down
        detections(1, 0) = 300.0f;                 // x1
        detections(1, 1) = 50.0f + frame * 8.0f;   // y1
        detections(1, 2) = 400.0f;                 // x2
        detections(1, 3) = 150.0f + frame * 8.0f; // y2
        detections(1, 4) = 0.75f;                  // conf
        detections(1, 5) = 1.0f;                   // cls
        
        // Object 3: moving diagonally
        detections(2, 0) = 450.0f + frame * 5.0f;  // x1
        detections(2, 1) = 200.0f + frame * 6.0f;  // y1
        detections(2, 2) = 550.0f + frame * 5.0f;  // x2
        detections(2, 3) = 300.0f + frame * 6.0f;  // y2
        detections(2, 4) = 0.7f;                   // conf
        detections(2, 5) = 2.0f;                   // cls
        
        // Update tracker
        Eigen::MatrixXf tracks = tracker.update(detections, img);
        
        std::cout << "Frame " << frame << ": Detected " << detections.rows() 
                  << " objects, Tracking " << tracks.rows() << " objects\n";
        
        if (tracks.rows() > 0) {
            for (int i = 0; i < tracks.rows(); ++i) {
                int id = static_cast<int>(tracks(i, 4));
                float conf = tracks(i, 5);
                int cls = static_cast<int>(tracks(i, 6));
                std::cout << "  Track ID: " << id 
                          << ", Class: " << cls 
                          << ", Confidence: " << conf << "\n";
            }
        }
        std::cout << "\n";
    }
    
    std::cout << "Test completed successfully!\n";
    return 0;
}

