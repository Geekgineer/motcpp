// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 motcpp contributors

#include <motcpp/config.hpp>
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <iostream>

namespace motcpp {

TrackerConfig load_tracker_config(const std::string& config_path) {
    TrackerConfig config;
    
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("Config file not found: " + config_path);
    }
    
    YAML::Node yaml_config = YAML::LoadFile(config_path);
    
    for (const auto& node : yaml_config) {
        std::string key = node.first.as<std::string>();
        const YAML::Node& value = node.second;
        
        if (value["type"]) {
            std::string type = value["type"].as<std::string>();
            
            if (type == "uniform" || type == "choice") {
                if (value["default"]) {
                    if (value["default"].IsScalar()) {
                        // Try to determine type from default value
                        std::string default_str = value["default"].as<std::string>();
                        try {
                            float fval = std::stof(default_str);
                            config.float_params[key] = fval;
                        } catch (...) {
                            // Not a float, try int
                            try {
                                int ival = std::stoi(default_str);
                                config.int_params[key] = ival;
                            } catch (...) {
                                config.string_params[key] = default_str;
                            }
                        }
                    }
                }
            }
        } else {
            // Direct value assignment
            if (value.IsScalar()) {
                try {
                    float fval = value.as<float>();
                    config.float_params[key] = fval;
                } catch (...) {
                    try {
                        int ival = value.as<int>();
                        config.int_params[key] = ival;
                    } catch (...) {
                        std::string sval = value.as<std::string>();
                        if (sval == "true" || sval == "True") {
                            config.bool_params[key] = true;
                        } else if (sval == "false" || sval == "False") {
                            config.bool_params[key] = false;
                        } else {
                            config.string_params[key] = sval;
                        }
                    }
                }
            }
        }
    }
    
    return config;
}

std::string get_tracker_config_path(const std::string& tracker_type) {
    // Default configs directory relative to executable or project root
    std::string base_path = "configs/trackers";
    return base_path + "/" + tracker_type + ".yaml";
}

} // namespace motcpp

