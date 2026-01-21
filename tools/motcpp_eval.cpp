#include <motcpp/trackers/sort.hpp>
#include <motcpp/trackers/ucmc.hpp>
#include <motcpp/trackers/bytetrack.hpp>
#include <motcpp/trackers/ocsort.hpp>
#include <motcpp/trackers/deepocsort.hpp>
#include <motcpp/trackers/strongsort.hpp>
#include <motcpp/trackers/botsort.hpp>
#include <motcpp/trackers/boosttrack.hpp>
#include <motcpp/trackers/hybridsort.hpp>
#include <motcpp/data/mot17_dataset.hpp>
#include <motcpp/utils/mot_format.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mot_root> <output_dir> [tracking_method] [det_emb_root] [model_name] [reid_name] [reid_weights]\n";
        std::cerr << "Example: " << argv[0] << " ../assets/MOT17-mini/train ./results bytetrack\n";
        std::cerr << "Example with pre-generated dets/embs: " << argv[0] 
                  << " <mot_root> <output_dir> bytetrack ../assets/yolox_x_ablation yolox_x_ablation osnet_x1_0_dukemtmcreid\n";
        std::cerr << "Example for DeepOCSort: " << argv[0]
                  << " <mot_root> <output_dir> deepocsort <det_emb_root> <model_name> <reid_name> <reid_weights.onnx>\n";
        std::cerr << "Example for StrongSORT: " << argv[0]
                  << " <mot_root> <output_dir> strongsort <det_emb_root> <model_name> <reid_name> <reid_weights.onnx>\n";
        return 1;
    }
    
    std::string mot_root = argv[1];
    std::string output_dir = argv[2];
    std::string tracking_method = (argc > 3) ? argv[3] : "bytetrack";
    std::string det_emb_root = (argc > 4) ? argv[4] : "";
    std::string model_name = (argc > 5) ? argv[5] : "";
    std::string reid_name = (argc > 6) ? argv[6] : "";
    std::string reid_weights = (argc > 7) ? argv[7] : "";
    
    std::cout << "motcpp - MOT Evaluation Tool v1.0.0\n";
    std::cout << "==========================\n\n";
    std::cout << "MOT Root: " << mot_root << "\n";
    std::cout << "Output Dir: " << output_dir << "\n";
    std::cout << "Tracking Method: " << tracking_method << "\n";
    if (!det_emb_root.empty()) {
        std::cout << "Det/Emb Root: " << det_emb_root << "\n";
        std::cout << "Model Name: " << model_name << "\n";
        std::cout << "ReID Name: " << reid_name << "\n";
    }
    std::cout << "\n";
    
    // Create dataset
    motcpp::data::MOT17Dataset dataset(mot_root, det_emb_root, model_name, reid_name);
    
    // Create output directory
    std::filesystem::create_directories(output_dir);
    
    // Process each sequence
    auto seq_names = dataset.sequence_names();
    std::cout << "Found " << seq_names.size() << " sequences\n\n";
    
    for (const auto& seq_name : seq_names) {
        std::cout << "Processing sequence: " << seq_name << "\n";
        
        try {
            auto seq_info = dataset.get_sequence_info(seq_name);
            
            // Load detections
            std::cout << "  Detection file path: " << seq_info.det_path << "\n";
            std::cout << "  File exists: " << (std::filesystem::exists(seq_info.det_path) ? "yes" : "no") << "\n";
            auto detections = dataset.load_detections(seq_info.det_path);
            std::cout << "  Loaded detections for " << detections.size() << " frames\n";
            
            // Load embeddings if available
            std::unordered_map<int, Eigen::MatrixXf> embeddings;
            if (!det_emb_root.empty() && !model_name.empty() && !reid_name.empty()) {
                std::filesystem::path emb_dir = std::filesystem::path(det_emb_root) / model_name / "embs" / reid_name;
                // Extract sequence number (e.g., "02" from "MOT17-02-FRCNN")
                std::string emb_filename;
                size_t first_dash = seq_name.find('-');
                size_t second_dash = seq_name.find('-', first_dash + 1);
                if (second_dash != std::string::npos) {
                    std::string seq_num = seq_name.substr(first_dash + 1, second_dash - first_dash - 1);
                    emb_filename = "MOT17-" + seq_num + ".txt";
                } else {
                    emb_filename = seq_name + ".txt";
                }
                std::filesystem::path emb_path = emb_dir / emb_filename;
                
                if (std::filesystem::exists(emb_path)) {
                    embeddings = dataset.load_embeddings(emb_path, detections);
                    std::cout << "  Loaded embeddings for " << embeddings.size() << " frames\n";
                }
            }
            
            // Initialize tracker based on tracking_method
            std::unique_ptr<motcpp::BaseTracker> tracker;
            
            if (tracking_method == "sort") {
                // SORT - Original Simple Online and Realtime Tracking
                tracker = std::make_unique<motcpp::trackers::Sort>(
                    0.3f,   // det_thresh
                    1,      // max_age (original SORT uses 1)
                    50,     // max_obs
                    3,      // min_hits
                    0.3f,   // iou_threshold
                    false,  // per_class
                    80,     // nr_classes
                    "iou",  // asso_func
                    false   // is_obb
                );
            } else if (tracking_method == "ucmc") {
                // UCMCTrack - Unified Confidence-based Multi-object tracker
                tracker = std::make_unique<motcpp::trackers::UCMCTrack>(
                    0.3f,   // det_thresh
                    30,     // max_age
                    50,     // max_obs
                    3,      // min_hits
                    0.3f,   // iou_threshold
                    false,  // per_class
                    80,     // nr_classes
                    "iou",  // asso_func
                    false,  // is_obb
                    100.0,  // a1 - high-conf association threshold
                    100.0,  // a2 - low-conf association threshold
                    5.0,    // wx - process noise x
                    5.0,    // wy - process noise y
                    10.0,   // vmax - max velocity
                    1.0 / seq_info.fps,  // dt - time step
                    0.5f    // high_score - confidence split
                );
            } else if (tracking_method == "bytetrack") {
                tracker = std::make_unique<motcpp::trackers::ByteTrack>(
                    0.3f,   // det_thresh
                    30,     // max_age
                    50,     // max_obs
                    3,      // min_hits
                    0.3f,   // iou_threshold
                    false,  // per_class
                    80,     // nr_classes
                    "iou",  // asso_func
                    false,  // is_obb
                    0.1f,   // min_conf
                    0.45f,  // track_thresh
                    0.8f,   // match_thresh
                    30,     // track_buffer
                    seq_info.fps  // frame_rate
                );
            } else if (tracking_method == "ocsort") {
                tracker = std::make_unique<motcpp::trackers::OCSort>(
                    0.2f,   // det_thresh (OCSort default)
                    30,     // max_age
                    50,     // max_obs
                    3,      // min_hits
                    0.3f,   // iou_threshold
                    false,  // per_class
                    80,     // nr_classes
                    "iou",  // asso_func
                    false,  // is_obb
                    0.1f,   // min_conf
                    3,      // delta_t
                    0.2f,   // inertia
                    false,  // use_byte
                    0.01f,  // Q_xy_scaling
                    0.0001f // Q_s_scaling
                );
            } else if (tracking_method == "deepocsort") {
                if (reid_weights.empty()) {
                    std::cerr << "Error: DeepOCSort requires reid_weights path (7th argument)\n";
                    return 1;
                }
                tracker = std::make_unique<motcpp::trackers::DeepOCSort>(
                    reid_weights,  // reid_weights
                    false,        // use_half
                    false,        // use_gpu
                    0.3f,         // det_thresh
                    30,           // max_age
                    50,           // max_obs
                    3,            // min_hits
                    0.3f,         // iou_threshold
                    false,        // per_class
                    80,           // nr_classes
                    "iou",        // asso_func
                    false,        // is_obb
                    3,            // delta_t
                    0.2f,         // inertia
                    0.5f,         // w_association_emb
                    0.95f,        // alpha_fixed_emb
                    0.5f,         // aw_param
                    false,        // embedding_off
                    false,        // cmc_off
                    false,        // aw_off
                    0.01f,        // Q_xy_scaling
                    0.0001f       // Q_s_scaling
                );
            } else if (tracking_method == "strongsort") {
                // ReID weights are optional - can use pre-generated embeddings instead
                // Default parameters for strongsort.yaml
                tracker = std::make_unique<motcpp::trackers::StrongSORT>(
                    reid_weights,  // reid_weights
                    false,         // use_half
                    false,         // use_gpu
                    0.3f,          // det_thresh
                    30,            // max_age
                    50,            // max_obs
                    3,             // min_hits
                    0.3f,          // iou_threshold
                    false,         // per_class
                    80,            // nr_classes
                    "iou",         // asso_func
                    false,         // is_obb
                    0.6f,          // min_conf (Python default: 0.6)
                    0.4f,          // max_cos_dist (Python default: 0.4)
                    0.7f,          // max_iou_dist
                    3,             // n_init
                    100,           // nn_budget
                    0.98f,         // mc_lambda
                    0.9f           // ema_alpha
                );
            } else if (tracking_method == "botsort") {
                // Default parameters for botsort.yaml
                tracker = std::make_unique<motcpp::trackers::BotSort>(
                    reid_weights.empty() ? "" : reid_weights,  // reid_weights
                    false,         // use_half
                    false,         // use_gpu
                    0.3f,          // det_thresh
                    30,            // max_age (track_buffer)
                    50,            // max_obs
                    3,             // min_hits
                    0.3f,          // iou_threshold
                    false,         // per_class
                    80,            // nr_classes
                    "iou",         // asso_func
                    false,         // is_obb
                    0.6f,          // track_high_thresh (Python default: 0.6)
                    0.1f,          // track_low_thresh (Python default: 0.1)
                    0.7f,          // new_track_thresh (Python default: 0.7)
                    30,            // track_buffer (Python default: 30)
                    0.8f,          // match_thresh (Python default: 0.8)
                    0.5f,          // proximity_thresh (Python default: 0.5)
                    0.25f,         // appearance_thresh (Python default: 0.25)
                    "ecc",         // cmc_method (Python default: ecc)
                    seq_info.fps,  // frame_rate
                    false,         // fuse_first_associate
                    !reid_weights.empty()  // with_reid
                );
            } else if (tracking_method == "boosttrack") {
                // Default parameters for boosttrack.yaml
                // BoostTrack++ uses use_rich_s=True, use_sb=True, use_vt=True
                tracker = std::make_unique<motcpp::trackers::BoostTrackTracker>(
                    reid_weights.empty() ? "" : reid_weights,  // reid_weights
                    false,         // use_half
                    false,         // use_gpu
                    0.6f,          // det_thresh (Python default: 0.6)
                    60,            // max_age (Python default: 60)
                    50,            // max_obs
                    3,             // min_hits (Python default: 3)
                    0.3f,          // iou_threshold (Python default: 0.3)
                    false,         // per_class
                    80,            // nr_classes
                    "iou",         // asso_func
                    false,         // is_obb
                    true,          // use_ecc (Python default: True)
                    10,            // min_box_area (Python default: 10)
                    1.6f,          // aspect_ratio_thresh (Python default: 1.6)
                    "ecc",         // cmc_method
                    0.5f,          // lambda_iou (Python default: 0.5)
                    0.25f,         // lambda_mhd (Python default: 0.25)
                    0.25f,         // lambda_shape (Python default: 0.25)
                    true,          // use_dlo_boost (Python default: True)
                    true,          // use_duo_boost (Python default: True)
                    0.65f,         // dlo_boost_coef (Python default: 0.65)
                    false,         // s_sim_corr (Python default: False)
                    true,          // use_rich_s (Python default: True for BoostTrack++)
                    true,          // use_sb (Python default: True for BoostTrack++)
                    true,          // use_vt (Python default: True for BoostTrack++)
                    !reid_weights.empty()  // with_reid (Python default: True)
                );
            } else if (tracking_method == "hybridsort") {
                // Default parameters for hybridsort.yaml
                tracker = std::make_unique<motcpp::trackers::HybridSort>(
                    reid_weights.empty() ? "" : reid_weights,  // reid_weights
                    false,         // use_half
                    false,         // use_gpu
                    0.5f,          // det_thresh (use track_thresh from Python: 0.5)
                    30,            // max_age
                    50,            // max_obs
                    3,             // min_hits
                    0.3f,          // iou_threshold (typical tracking threshold)
                    false,         // per_class
                    80,            // nr_classes
                    "hmiou",       // asso_func
                    false,         // is_obb
                    0.1f,          // low_thresh (Python default: 0.1)
                    3,             // delta_t (Python default: 3)
                    0.05f,         // inertia (Python default: 0.05)
                    true,          // use_byte (Python default: True)
                    true,          // use_custom_kf (Python default: True)
                    30,            // longterm_bank_length (Python default: 30)
                    0.9f,          // alpha (Python default: 0.9)
                    false,         // adapfs (Python default: False)
                    0.5f,          // track_thresh (Python default: 0.5)
                    4.6f,          // EG_weight_high_score (Python default: 4.6)
                    1.3f,          // EG_weight_low_score (Python default: 1.3)
                    true,          // TCM_first_step (Python default: True)
                    true,          // TCM_byte_step (Python default: True)
                    1.0f,          // TCM_byte_step_weight (Python default: 1.0)
                    0.7f,          // high_score_matching_thresh (Python default: 0.7)
                    true,          // with_longterm_reid (Python default: True)
                    0.0f,          // longterm_reid_weight (Python default: 0.0)
                    true,          // with_longterm_reid_correction (Python default: True)
                    0.4f,          // longterm_reid_correction_thresh (Python default: 0.4)
                    0.4f,          // longterm_reid_correction_thresh_low (Python default: 0.4)
                    "ecc",         // cmc_method
                    !reid_weights.empty()  // with_reid
                );
            } else {
                std::cerr << "Unknown tracking method: " << tracking_method << "\n";
                std::cerr << "Supported methods: sort, ucmc, bytetrack, ocsort, deepocsort, strongsort, botsort, boosttrack, hybridsort\n";
                return 1;
            }
            
            // Output file for this sequence
            std::filesystem::path output_file = std::filesystem::path(output_dir) / (seq_name + ".txt");
            if (std::filesystem::exists(output_file)) {
                std::filesystem::remove(output_file);
            }
            
            // Process frames - use all frames with detections, not just frames with images
            std::vector<int> frames_to_process;
            for (const auto& [frame_id, dets] : detections) {
                frames_to_process.push_back(frame_id);
            }
            std::sort(frames_to_process.begin(), frames_to_process.end());
            
            // Detect ablation dataset: if GT only covers second half of detections
            // MOT17-ablation: det frames 1-600, GT frames 1-299 (offset=300)
            int frame_offset = 0;
            if (!frames_to_process.empty() && std::filesystem::exists(seq_info.gt_path)) {
                // Read GT to find max frame
                std::ifstream gt_file(seq_info.gt_path);
                std::string line;
                int max_gt_frame = 0;
                while (std::getline(gt_file, line)) {
                    if (line.empty() || line[0] == '#') continue;
                    std::istringstream iss(line);
                    std::string token;
                    std::getline(iss, token, ',');
                    int gt_frame = std::stoi(token);
                    max_gt_frame = std::max(max_gt_frame, gt_frame);
                }
                
                int max_det_frame = frames_to_process.back();
                int min_det_frame = frames_to_process.front();
                
                // If det frames go beyond GT frames (e.g., det 1-600, GT 1-299)
                // This indicates ablation dataset where det frame 301+ maps to GT frame 1+
                if (max_det_frame > max_gt_frame * 1.5 && max_gt_frame > 0) {
                    frame_offset = max_det_frame - max_gt_frame;
                    std::cout << "  Detected ablation offset: " << frame_offset << " (det " << min_det_frame << "-" << max_det_frame 
                              << " -> GT 1-" << max_gt_frame << ")\n";
                    
                    // Filter to only process frames that map to GT range
                    std::vector<int> filtered_frames;
                    for (int f : frames_to_process) {
                        if (f > frame_offset) {
                            filtered_frames.push_back(f);
                        }
                    }
                    frames_to_process = filtered_frames;
                    std::cout << "  Processing frames " << frames_to_process.front() << "-" << frames_to_process.back() 
                              << " (" << frames_to_process.size() << " frames)\n";
                }
            }
            
            int processed_frames = 0;
            for (int frame_id : frames_to_process) {
                try {
                    // Create a dummy image if we don't have one for this frame
                    cv::Mat dummy_img = cv::Mat::zeros(1080, 1920, CV_8UC3);
                    
                    // Try to get real image if available
                    auto frame_data = dataset.get_frame(seq_info, frame_id, detections, embeddings);
                    cv::Mat img = frame_data.image;
                    
                    // Get detections and embeddings for this frame
                    Eigen::MatrixXf frame_dets = Eigen::MatrixXf(0, 6);
                    Eigen::MatrixXf frame_embs = Eigen::MatrixXf(0, 0);
                    
                    auto det_it = detections.find(frame_id);
                    if (det_it != detections.end()) {
                        frame_dets = det_it->second;
                    }
                    
                    auto emb_it = embeddings.find(frame_id);
                    if (emb_it != embeddings.end()) {
                        frame_embs = emb_it->second;
                    } else if (frame_dets.rows() > 0) {
                        frame_embs = Eigen::MatrixXf(frame_dets.rows(), 0);
                    }
                    
                    // Run tracker
                    Eigen::MatrixXf tracks = tracker->update(frame_dets, img, frame_embs);
                    
                    // Convert to MOT format and write
                    // Apply frame offset for ablation datasets
                    int output_frame_id = (frame_offset > 0) ? (frame_id - frame_offset) : frame_id;
                    if (tracks.rows() > 0) {
                        Eigen::MatrixXf mot_results = motcpp::utils::convert_to_mot_format(tracks, output_frame_id);
                        motcpp::utils::write_mot_results(output_file, mot_results);
                    }
                    
                    processed_frames++;
                } catch (const std::exception& e) {
                    // If frame doesn't have image, use dummy image
                    try {
                        Eigen::MatrixXf frame_dets = Eigen::MatrixXf(0, 6);
                        Eigen::MatrixXf frame_embs = Eigen::MatrixXf(0, 0);
                        
                        auto det_it = detections.find(frame_id);
                        if (det_it != detections.end()) {
                            frame_dets = det_it->second;
                        }
                        
                        auto emb_it = embeddings.find(frame_id);
                        if (emb_it != embeddings.end()) {
                            frame_embs = emb_it->second;
                        } else if (frame_dets.rows() > 0) {
                            frame_embs = Eigen::MatrixXf(frame_dets.rows(), 0);
                        }
                        
                        cv::Mat dummy_img = cv::Mat::zeros(1080, 1920, CV_8UC3);
                        Eigen::MatrixXf tracks = tracker->update(frame_dets, dummy_img, frame_embs);
                        
                        // Apply frame offset for ablation datasets
                        int output_frame_id2 = (frame_offset > 0) ? (frame_id - frame_offset) : frame_id;
                        if (tracks.rows() > 0) {
                            Eigen::MatrixXf mot_results = motcpp::utils::convert_to_mot_format(tracks, output_frame_id2);
                            motcpp::utils::write_mot_results(output_file, mot_results);
                        }
                        
                        processed_frames++;
                    } catch (const std::exception& e2) {
                        std::cerr << "  Error processing frame " << frame_id << ": " << e2.what() << "\n";
                    }
                }
            }
            
            std::cout << "  Processed " << processed_frames << " frames\n";
            std::cout << "  Results saved to: " << output_file << "\n\n";
            
            // Reset tracker for next sequence
            tracker->reset();
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing sequence " << seq_name << ": " << e.what() << "\n\n";
        }
    }
    
    std::cout << "Evaluation completed!\n";
    std::cout << "Results saved to: " << output_dir << "\n";
    std::cout << "\nTo compute metrics, use TrackEval Python script:\n";
    std::filesystem::path script_path = std::filesystem::path(argv[0]).parent_path().parent_path() / "scripts" / "eval_mot.py";
    std::cout << "  python3 " << script_path << " --gt_folder " << mot_root 
              << " --trackers_folder " << output_dir << "\n";
    
    return 0;
}

