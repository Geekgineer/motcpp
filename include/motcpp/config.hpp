// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <yaml-cpp/yaml.h>

namespace motcpp {

/**
 * Tracker configuration parameters
 */
struct TrackerConfig {
    std::unordered_map<std::string, float> float_params;
    std::unordered_map<std::string, int> int_params;
    std::unordered_map<std::string, bool> bool_params;
    std::unordered_map<std::string, std::string> string_params;
    
    float get_float(const std::string& key, float default_value) const {
        auto it = float_params.find(key);
        return (it != float_params.end()) ? it->second : default_value;
    }
    
    int get_int(const std::string& key, int default_value) const {
        auto it = int_params.find(key);
        return (it != int_params.end()) ? it->second : default_value;
    }
    
    bool get_bool(const std::string& key, bool default_value) const {
        auto it = bool_params.find(key);
        return (it != bool_params.end()) ? it->second : default_value;
    }
    
    std::string get_string(const std::string& key, const std::string& default_value) const {
        auto it = string_params.find(key);
        return (it != string_params.end()) ? it->second : default_value;
    }
};

/**
 * Load tracker configuration from YAML file
 */
TrackerConfig load_tracker_config(const std::string& config_path);

/**
 * Get default config path for a tracker type
 */
std::string get_tracker_config_path(const std::string& tracker_type);

} // namespace motcpp

