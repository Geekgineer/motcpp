#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 motcpp contributors
#
# Prepare benchmark data for GitHub Release
# Run this to create archives that can be uploaded to releases

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_ROOT}/release_assets"
ASSETS_DIR="${PROJECT_ROOT}/assets"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=== Preparing Benchmark Data for Release ==="
echo ""

# Check if assets directory exists
if [ ! -d "$ASSETS_DIR" ]; then
    echo -e "${YELLOW}Warning: assets/ directory not found${NC}"
    echo ""
    echo "To prepare benchmark data, you need to:"
    echo ""
    echo "1. Download MOT17 dataset (mini version for testing):"
    echo "   mkdir -p assets/MOT17-mini/train"
    echo "   # Copy sequences: MOT17-02-FRCNN, MOT17-04-FRCNN, etc."
    echo ""
    echo "2. Generate or copy YOLOX detections:"
    echo "   mkdir -p assets/yolox_x_ablation/dets"
    echo "   # Format: frame,id,x,y,w,h,conf,-1,-1,-1"
    echo ""
    echo "3. (Optional) Generate ReID embeddings:"
    echo "   mkdir -p assets/reid_embs"
    echo ""
    echo "Or copy from boxmot-cpp assets if available:"
    echo "   cp -r ../boxmot-cpp/assets/* assets/"
    echo ""
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

CREATED_FILES=0

# 1. Package MOT17-mini (sample sequences)
if [ -d "assets/MOT17-mini" ]; then
    echo "Packaging MOT17-mini..."
    tar -czf "${OUTPUT_DIR}/MOT17-mini.tar.gz" -C assets MOT17-mini
    echo -e "  ${GREEN}✓${NC} Created: MOT17-mini.tar.gz ($(du -h "${OUTPUT_DIR}/MOT17-mini.tar.gz" | cut -f1))"
    CREATED_FILES=$((CREATED_FILES + 1))
else
    echo -e "  ${YELLOW}⚠${NC} Skipped: MOT17-mini (not found)"
fi

# 2. Package detection files
if [ -d "assets/yolox_x_ablation" ]; then
    echo "Packaging YOLOX detections..."
    tar -czf "${OUTPUT_DIR}/yolox_dets.tar.gz" -C assets yolox_x_ablation
    echo -e "  ${GREEN}✓${NC} Created: yolox_dets.tar.gz ($(du -h "${OUTPUT_DIR}/yolox_dets.tar.gz" | cut -f1))"
    CREATED_FILES=$((CREATED_FILES + 1))
else
    echo -e "  ${YELLOW}⚠${NC} Skipped: yolox_dets (not found)"
fi

# 3. Package ReID embeddings (if available)
# Check both standalone reid_embs and yolox_x_ablation/embs
if [ -d "assets/reid_embs" ]; then
    echo "Packaging ReID embeddings (standalone)..."
    tar -czf "${OUTPUT_DIR}/reid_embs.tar.gz" -C assets reid_embs
    echo -e "  ${GREEN}✓${NC} Created: reid_embs.tar.gz ($(du -h "${OUTPUT_DIR}/reid_embs.tar.gz" | cut -f1))"
    CREATED_FILES=$((CREATED_FILES + 1))
elif [ -d "assets/yolox_x_ablation/embs" ]; then
    echo "Packaging ReID embeddings (from yolox_x_ablation/embs)..."
    tar -czf "${OUTPUT_DIR}/reid_embs.tar.gz" -C assets/yolox_x_ablation embs
    echo -e "  ${GREEN}✓${NC} Created: reid_embs.tar.gz ($(du -h "${OUTPUT_DIR}/reid_embs.tar.gz" | cut -f1))"
    CREATED_FILES=$((CREATED_FILES + 1))
else
    echo -e "  ${YELLOW}⚠${NC} Skipped: reid_embs (not found)"
fi

# Check if any files were created
if [ $CREATED_FILES -eq 0 ]; then
    echo ""
    echo -e "${RED}Error: No data files found to package!${NC}"
    echo ""
    echo "Please ensure you have benchmark data in the assets/ directory."
    echo "See above for required directory structure."
    exit 1
fi

# 4. Create checksums
echo ""
echo "Creating checksums..."
cd "$OUTPUT_DIR"
if ls *.tar.gz 1> /dev/null 2>&1; then
    sha256sum *.tar.gz > SHA256SUMS.txt
    cat SHA256SUMS.txt
else
    echo -e "${RED}No tar.gz files found to create checksums${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Release Assets Ready ===${NC}"
echo "Location: $OUTPUT_DIR"
echo ""
echo "Created $CREATED_FILES archive(s):"
ls -la "$OUTPUT_DIR"/*.tar.gz 2>/dev/null || true
echo ""
echo "GitHub Release Instructions:"
echo "1. Go to https://github.com/Geekgineer/motcpp/releases"
echo "2. Create new release with tag 'benchmark-data-v1.0'"
echo "3. Upload all files from $OUTPUT_DIR"
echo "4. Users can then run: ./scripts/auto_benchmark.sh --all"
