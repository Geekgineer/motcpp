#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 motcpp contributors
# https://github.com/Geekgineer/motcpp
#
# Auto Benchmark Script for motcpp
# Downloads benchmark data from GitHub Releases and runs evaluation

set -e

# Configuration
REPO_URL="https://github.com/Geekgineer/motcpp"
RELEASE_TAG="benchmark-data-v1.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/benchmark_data"
BUILD_DIR="${PROJECT_ROOT}/build"
RESULTS_DIR="${PROJECT_ROOT}/benchmark_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Print banner
print_banner() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}                   ${GREEN}motcpp Auto Benchmark${NC}                    ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}          Modern C++ Multi-Object Tracking Library          ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}            https://github.com/Geekgineer/motcpp            ${BLUE}║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Check dependencies
check_dependencies() {
    info "Checking dependencies..."
    
    local missing=()
    
    command -v curl >/dev/null 2>&1 || missing+=("curl")
    command -v tar >/dev/null 2>&1 || missing+=("tar")
    command -v cmake >/dev/null 2>&1 || missing+=("cmake")
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    
    if [ ${#missing[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing[*]}\nPlease install them and try again."
    fi
    
    success "All dependencies found"
}

# Download benchmark data from GitHub Releases
download_benchmark_data() {
    info "Downloading benchmark data from GitHub Releases..."
    
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"
    
    # Files to download from release
    local files=(
        "MOT17-mini.tar.gz"
        "yolox_dets.tar.gz"
        "reid_embs.tar.gz"
    )
    
    for file in "${files[@]}"; do
        local url="${REPO_URL}/releases/download/${RELEASE_TAG}/${file}"
        
        if [ -f "$file" ]; then
            info "  $file already exists, skipping..."
        else
            info "  Downloading $file..."
            if curl -fsSL -o "$file" "$url"; then
                success "  Downloaded $file"
            else
                warning "  Failed to download $file (may not exist in release)"
            fi
        fi
    done
    
    # Extract archives
    info "Extracting data..."
    for file in *.tar.gz; do
        if [ -f "$file" ]; then
            tar -xzf "$file" 2>/dev/null || true
            info "  Extracted $file"
        fi
    done
    
    success "Benchmark data ready at $DATA_DIR"
}

# Build motcpp
build_project() {
    info "Building motcpp..."
    
    cd "$PROJECT_ROOT"
    
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
    fi
    
    cd "$BUILD_DIR"
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DMOTCPP_BUILD_TESTS=ON \
        -DMOTCPP_BUILD_TOOLS=ON \
        -DMOTCPP_ENABLE_ONNX=ON
    
    cmake --build . -j$(nproc)
    
    success "Build complete"
}

# Run unit tests
run_tests() {
    info "Running unit tests..."
    
    cd "$BUILD_DIR"
    ctest --output-on-failure
    
    success "All tests passed"
}

# Run tracking benchmark
run_benchmark() {
    local tracker=$1
    local det_name=$2
    
    info "Running benchmark: $tracker with $det_name detections..."
    
    local result_dir="${RESULTS_DIR}/${tracker}_${det_name}"
    mkdir -p "$result_dir"
    
    cd "$BUILD_DIR"
    
    # Check if data exists
    if [ ! -d "${DATA_DIR}/MOT17-mini" ]; then
        warning "MOT17-mini not found, using sample data..."
        # Use project's built-in sample data if available
        local sample_data="${PROJECT_ROOT}/assets/MOT17-mini"
        if [ -d "$sample_data" ]; then
            ln -sf "$sample_data" "${DATA_DIR}/MOT17-mini"
        else
            error "No benchmark data available. Please download from releases."
        fi
    fi
    
    # Run evaluation
    ./tools/motcpp_eval \
        --tracker "$tracker" \
        --dataset "${DATA_DIR}/MOT17-mini/train" \
        --dets "$det_name" \
        --output "$result_dir" \
        2>&1 | tee "${result_dir}/log.txt"
    
    success "Benchmark complete for $tracker"
}

# Run TrackEval
run_trackeval() {
    info "Running TrackEval metrics..."
    
    cd "$PROJECT_ROOT"
    
    # Setup Python environment if needed
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        source .venv/bin/activate
        pip install -q trackeval numpy
    else
        source .venv/bin/activate
    fi
    
    # Run TrackEval
    python3 -m trackeval.eval_mot \
        --GT_FOLDER "${DATA_DIR}/MOT17-mini/train" \
        --TRACKERS_FOLDER "$RESULTS_DIR" \
        --BENCHMARK MOT17 \
        --SPLIT_TO_EVAL train \
        --METRICS HOTA CLEAR Identity \
        --USE_PARALLEL True \
        2>&1 | tee "${RESULTS_DIR}/trackeval_results.txt"
    
    success "TrackEval complete"
}

# Generate benchmark report
generate_report() {
    info "Generating benchmark report..."
    
    local report_file="${RESULTS_DIR}/benchmark_report.md"
    
    cat > "$report_file" << EOF
# motcpp Benchmark Report

**Date**: $(date '+%Y-%m-%d %H:%M:%S')
**System**: $(uname -s) $(uname -r) $(uname -m)
**CPU**: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 || echo "N/A")

## Trackers Evaluated

| Tracker | Type | Status |
|---------|------|--------|
EOF

    for tracker in sort bytetrack ocsort ucmctrack; do
        local status="✅ Complete"
        if [ ! -d "${RESULTS_DIR}/${tracker}_yolox" ]; then
            status="❌ Not run"
        fi
        echo "| $tracker | Motion | $status |" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF

## Results

See individual result directories in \`benchmark_results/\` for detailed metrics.

### TrackEval Output

\`\`\`
$(cat "${RESULTS_DIR}/trackeval_results.txt" 2>/dev/null || echo "TrackEval not run")
\`\`\`

## How to Reproduce

\`\`\`bash
./scripts/auto_benchmark.sh --all
\`\`\`

---
Generated by motcpp auto_benchmark.sh
EOF

    success "Report generated: $report_file"
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --download        Download benchmark data from GitHub Releases"
    echo "  --build           Build the project"
    echo "  --test            Run unit tests"
    echo "  --benchmark       Run tracking benchmarks"
    echo "  --eval            Run TrackEval metrics"
    echo "  --report          Generate benchmark report"
    echo "  --all             Run all steps"
    echo "  --tracker NAME    Specify tracker (default: all motion trackers)"
    echo "  --clean           Clean build and results directories"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Trackers available:"
    echo "  Motion-only: sort, bytetrack, ocsort, ucmctrack"
    echo "  ReID-based:  deepocsort, strongsort, botsort, boosttrack, hybridsort"
    echo ""
    echo "Examples:"
    echo "  $0 --all                    # Full benchmark pipeline"
    echo "  $0 --benchmark --tracker bytetrack"
    echo "  $0 --download --build"
}

# Clean directories
clean() {
    info "Cleaning build and results directories..."
    rm -rf "$BUILD_DIR" "$RESULTS_DIR"
    success "Clean complete"
}

# Main
main() {
    print_banner
    check_dependencies
    
    local do_download=false
    local do_build=false
    local do_test=false
    local do_benchmark=false
    local do_eval=false
    local do_report=false
    local tracker="all"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --download) do_download=true; shift ;;
            --build) do_build=true; shift ;;
            --test) do_test=true; shift ;;
            --benchmark) do_benchmark=true; shift ;;
            --eval) do_eval=true; shift ;;
            --report) do_report=true; shift ;;
            --all)
                do_download=true
                do_build=true
                do_test=true
                do_benchmark=true
                do_eval=true
                do_report=true
                shift
                ;;
            --tracker) tracker=$2; shift 2 ;;
            --clean) clean; exit 0 ;;
            -h|--help) usage; exit 0 ;;
            *) error "Unknown option: $1\nUse --help for usage information" ;;
        esac
    done
    
    # Default to --help if no options
    if ! $do_download && ! $do_build && ! $do_test && ! $do_benchmark && ! $do_eval && ! $do_report; then
        usage
        exit 0
    fi
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Run steps
    $do_download && download_benchmark_data
    $do_build && build_project
    $do_test && run_tests
    
    if $do_benchmark; then
        if [ "$tracker" = "all" ]; then
            for t in sort bytetrack ocsort ucmctrack; do
                run_benchmark "$t" "yolox"
            done
        else
            run_benchmark "$tracker" "yolox"
        fi
    fi
    
    $do_eval && run_trackeval
    $do_report && generate_report
    
    echo ""
    success "All tasks completed!"
    echo ""
    echo "Results: $RESULTS_DIR"
    echo "Report:  ${RESULTS_DIR}/benchmark_report.md"
}

main "$@"
