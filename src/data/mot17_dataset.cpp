// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/data/mot17_dataset.hpp>
#include <motcpp/utils/ops.hpp>
#include <algorithm>
#include <regex>
#include <sstream>

namespace motcpp::data {

MOT17Dataset::MOT17Dataset(const std::string& mot_root, 
                         const std::string& det_emb_root,
                         const std::string& model_name,
                         const std::string& /* reid_name */)
    : mot_root_(mot_root)
{
    if (!det_emb_root.empty() && !model_name.empty()) {
        // det_emb_root should be like "../assets/yolox_x_ablation"
        // Check if dets folder exists directly under det_emb_root
        std::filesystem::path test_path = std::filesystem::path(det_emb_root) / "dets";
        if (std::filesystem::exists(test_path)) {
            det_path_ = test_path;
        } else {
            // Otherwise, assume structure: det_emb_root/model_name/dets
            det_path_ = std::filesystem::path(det_emb_root) / model_name / "dets";
        }
    }
    index_sequences();
}

void MOT17Dataset::index_sequences() {
    if (!std::filesystem::exists(mot_root_)) {
        throw std::runtime_error("MOT root directory does not exist: " + mot_root_.string());
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(mot_root_)) {
        if (!entry.is_directory()) continue;
        
        std::string seq_name = entry.path().filename().string();
        std::filesystem::path seq_dir = entry.path();
        std::filesystem::path img_dir = seq_dir / "img1";
            std::filesystem::path det_file;
            if (det_path_.empty()) {
                det_file = seq_dir / "det" / "det.txt";
            } else {
                // Try to find detection file matching sequence name
                // Sequence names are like "MOT17-02-FRCNN", detection files are "MOT17-02.txt"
                std::string det_filename;
                
                // Extract sequence number (e.g., "02" from "MOT17-02-FRCNN")
                size_t first_dash = seq_name.find('-');
                size_t second_dash = seq_name.find('-', first_dash + 1);
                if (second_dash != std::string::npos) {
                    std::string seq_num = seq_name.substr(first_dash + 1, second_dash - first_dash - 1);
                    det_filename = "MOT17-" + seq_num + ".txt";
                } else {
                    det_filename = seq_name + ".txt";
                }
                
                det_file = det_path_ / det_filename;
                
                // Fallback to sequence name if not found
                if (!std::filesystem::exists(det_file)) {
                    det_file = det_path_ / (seq_name + ".txt");
                }
            }
        std::filesystem::path gt_file = seq_dir / "gt" / "gt.txt";
        
        if (!std::filesystem::exists(img_dir)) continue;
        
        // Collect frame IDs and paths
        std::vector<int> frame_ids;
        std::vector<std::filesystem::path> frame_paths;
        for (const auto& img_entry : std::filesystem::directory_iterator(img_dir)) {
            if (img_entry.path().extension() == ".jpg" || 
                img_entry.path().extension() == ".png") {
                std::string stem = img_entry.path().stem().string();
                try {
                    int fid = std::stoi(stem);
                    frame_ids.push_back(fid);
                    frame_paths.push_back(img_entry.path());
                } catch (...) {
                    continue;
                }
            }
        }
        
        std::sort(frame_ids.begin(), frame_ids.end());
        std::sort(frame_paths.begin(), frame_paths.end(), 
                 [](const auto& a, const auto& b) {
                     return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
                 });
        
        SequenceInfo seq_info;
        seq_info.name = seq_name;
        seq_info.seq_dir = seq_dir;
        seq_info.img_dir = img_dir;
        seq_info.det_path = det_file;
        seq_info.gt_path = gt_file;
        seq_info.frame_ids = frame_ids;
        seq_info.frame_paths = frame_paths;
        seq_info.fps = read_seq_fps(seq_dir);
        
        sequences_.push_back(seq_info);
    }
    
    std::sort(sequences_.begin(), sequences_.end(),
             [](const auto& a, const auto& b) { return a.name < b.name; });
}

int MOT17Dataset::read_seq_fps(const std::filesystem::path& seq_dir) const {
    std::filesystem::path cfg_file = seq_dir / "seqinfo.ini";
    if (!std::filesystem::exists(cfg_file)) {
        return 30; // Default FPS
    }
    
    std::ifstream file(cfg_file);
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("frameRate") != std::string::npos) {
            std::regex fps_regex(R"(frameRate\s*=\s*(\d+))");
            std::smatch match;
            if (std::regex_search(line, match, fps_regex)) {
                return std::stoi(match[1].str());
            }
        }
    }
    return 30; // Default
}

std::vector<std::string> MOT17Dataset::sequence_names() const {
    std::vector<std::string> names;
    for (const auto& seq : sequences_) {
        names.push_back(seq.name);
    }
    return names;
}

SequenceInfo MOT17Dataset::get_sequence_info(const std::string& seq_name) const {
    for (const auto& seq : sequences_) {
        if (seq.name == seq_name) {
            return seq;
        }
    }
    throw std::runtime_error("Sequence not found: " + seq_name);
}

std::unordered_map<int, Eigen::MatrixXf> MOT17Dataset::load_detections(
    const std::filesystem::path& det_path) const {
    std::unordered_map<int, Eigen::MatrixXf> detections_map;
    
    if (!std::filesystem::exists(det_path)) {
        return detections_map;
    }
    
    std::ifstream file(det_path);
    std::string line;
    bool is_comma_separated = false;
    
    // Check first line to determine format
    if (std::getline(file, line)) {
        if (line.find(',') != std::string::npos) {
            is_comma_separated = true;
        }
        file.seekg(0); // Reset to beginning
    }
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::vector<float> values;
        std::istringstream iss(line);
        std::string token;
        
        if (is_comma_separated) {
            // MOT17 format: frame_id, -1, x1, y1, w, h, conf
            while (std::getline(iss, token, ',')) {
                try {
                    values.push_back(std::stof(token));
                } catch (...) {
                    break;
                }
            }
            
            if (values.size() < 7) continue;
            
            int frame_id = static_cast<int>(values[0]);
            float x1 = values[2];
            float y1 = values[3];
            float w = values[4];
            float h = values[5];
            float conf = values[6];
            float cls = (values.size() > 7) ? values[7] : 0.0f;
            
            // Convert tlwh to xyxy
            float x2 = x1 + w;
            float y2 = y1 + h;
            
            Eigen::VectorXf det(6);
            det << x1, y1, x2, y2, conf, cls;
            
            if (detections_map.find(frame_id) == detections_map.end()) {
                detections_map[frame_id] = Eigen::MatrixXf(0, 6);
            }
            
            int rows = detections_map[frame_id].rows();
            detections_map[frame_id].conservativeResize(rows + 1, 6);
            detections_map[frame_id].row(rows) = det.transpose();
        } else {
            // Pre-generated format: frame_id x1 y1 x2 y2 conf cls (space-separated)
            float val;
            while (iss >> val) {
                values.push_back(val);
            }
            
            if (values.size() < 7) continue;
            
            int frame_id = static_cast<int>(values[0]);
            float x1 = values[1];
            float y1 = values[2];
            float x2 = values[3];
            float y2 = values[4];
            float conf = values[5];
            float cls = values[6];
            
            Eigen::VectorXf det(6);
            det << x1, y1, x2, y2, conf, cls;
            
            if (detections_map.find(frame_id) == detections_map.end()) {
                detections_map[frame_id] = Eigen::MatrixXf(0, 6);
            }
            
            int rows = detections_map[frame_id].rows();
            detections_map[frame_id].conservativeResize(rows + 1, 6);
            detections_map[frame_id].row(rows) = det.transpose();
        }
    }
    
    return detections_map;
}

std::unordered_map<int, Eigen::MatrixXf> MOT17Dataset::load_embeddings(
    const std::filesystem::path& emb_path,
    const std::unordered_map<int, Eigen::MatrixXf>& detections) const {
    std::unordered_map<int, Eigen::MatrixXf> embeddings_map;
    
    if (!std::filesystem::exists(emb_path)) {
        return embeddings_map;
    }
    
    // Build a mapping from detection index to frame_id
    std::vector<std::pair<int, int>> det_frame_map; // (frame_id, detection_index_in_frame)
    for (const auto& [frame_id, dets] : detections) {
        for (int i = 0; i < dets.rows(); ++i) {
            det_frame_map.push_back({frame_id, i});
        }
    }
    
    std::ifstream file(emb_path);
    std::string line;
    int emb_idx = 0;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        if (emb_idx >= static_cast<int>(det_frame_map.size())) break;
        
        auto [frame_id, det_idx] = det_frame_map[emb_idx];
        
        std::vector<float> emb_values;
        std::istringstream iss(line);
        float val;
        while (iss >> val) {
            emb_values.push_back(val);
        }
        
        if (emb_values.empty()) continue;
        
        Eigen::VectorXf emb = Eigen::Map<Eigen::VectorXf>(emb_values.data(), emb_values.size());
        
        if (embeddings_map.find(frame_id) == embeddings_map.end()) {
            embeddings_map[frame_id] = Eigen::MatrixXf(0, emb.size());
        }
        
        int rows = embeddings_map[frame_id].rows();
        embeddings_map[frame_id].conservativeResize(rows + 1, emb.size());
        embeddings_map[frame_id].row(rows) = emb.transpose();
        
        emb_idx++;
    }
    
    return embeddings_map;
}

Eigen::Vector4f MOT17Dataset::tlwh2xyxy(const Eigen::Vector4f& tlwh) {
    float t = tlwh(0), l = tlwh(1), w = tlwh(2), h = tlwh(3);
    return Eigen::Vector4f(t, l, t + w, l + h);
}

Eigen::Vector4f MOT17Dataset::xyxy2tlwh(const Eigen::Vector4f& xyxy) {
    float x1 = xyxy(0), y1 = xyxy(1), x2 = xyxy(2), y2 = xyxy(3);
    return Eigen::Vector4f(x1, y1, x2 - x1, y2 - y1);
}

FrameData MOT17Dataset::get_frame(const SequenceInfo& seq_info, int frame_id,
                                 const std::unordered_map<int, Eigen::MatrixXf>& detections,
                                 const std::unordered_map<int, Eigen::MatrixXf>& embeddings) const {
    FrameData frame_data;
    frame_data.frame_id = frame_id;
    
    // Find frame path
    auto it = std::find(seq_info.frame_ids.begin(), seq_info.frame_ids.end(), frame_id);
    if (it == seq_info.frame_ids.end()) {
        throw std::runtime_error("Frame ID not found: " + std::to_string(frame_id));
    }
    size_t idx = std::distance(seq_info.frame_ids.begin(), it);
    std::filesystem::path img_path = seq_info.frame_paths[idx];
    
    // Load image
    frame_data.image = cv::imread(img_path.string());
    if (frame_data.image.empty()) {
        throw std::runtime_error("Failed to load image: " + img_path.string());
    }
    
    // Get detections for this frame
    auto det_it = detections.find(frame_id);
    if (det_it != detections.end()) {
        frame_data.detections = det_it->second;
    } else {
        frame_data.detections = Eigen::MatrixXf(0, 6);
    }
    
    // Get embeddings for this frame
    auto emb_it = embeddings.find(frame_id);
    if (emb_it != embeddings.end()) {
        frame_data.embeddings = emb_it->second;
    } else {
        // Create empty embeddings matching detection count
        int num_dets = frame_data.detections.rows();
        frame_data.embeddings = Eigen::MatrixXf(num_dets, 0);
    }
    
    return frame_data;
}

} // namespace motcpp::data

