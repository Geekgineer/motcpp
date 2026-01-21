# Benchmarking Guide

This guide explains how to benchmark motcpp trackers on the MOT (Multi-Object Tracking) dataset.

## Quick Start

The easiest way to run benchmarks is using the auto-benchmark script:

```bash
./scripts/auto_benchmark.sh --all
```

This will:
1. Download benchmark data from GitHub Releases
2. Build the project
3. Run unit tests
4. Execute tracking benchmarks
5. Calculate metrics with TrackEval
6. Generate a benchmark report

## Manual Benchmarking

### Step 1: Download Data

Benchmark data is hosted in GitHub Releases to keep the repository lightweight.

```bash
# Download from releases
./scripts/auto_benchmark.sh --download

# Or manually download
wget https://github.com/Geekgineer/motcpp/releases/download/benchmark-data-v1.0/MOT17-mini.tar.gz
wget https://github.com/Geekgineer/motcpp/releases/download/benchmark-data-v1.0/yolox_dets.tar.gz
tar -xzf MOT17-mini.tar.gz -C benchmark_data/
tar -xzf yolox_dets.tar.gz -C benchmark_data/
```

### Step 2: Build the Project

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMOTCPP_BUILD_TOOLS=ON
cmake --build build -j$(nproc)
```

### Step 3: Run Tracker

```bash
./build/tools/motcpp_eval \
    --tracker bytetrack \
    --dataset benchmark_data/MOT17-mini/train \
    --dets yolox \
    --output results/bytetrack_yolox/
```

### Step 4: Evaluate with TrackEval

```bash
python3 -m pip install trackeval
python3 -m trackeval.eval_mot \
    --GT_FOLDER benchmark_data/MOT17-mini/train \
    --TRACKERS_FOLDER results/ \
    --BENCHMARK MOT17 \
    --METRICS HOTA CLEAR Identity
```

## Available Trackers

| Tracker | Command | Type |
|---------|---------|------|
| SORT | `--tracker sort` | Motion |
| ByteTrack | `--tracker bytetrack` | Motion |
| OC-SORT | `--tracker ocsort` | Motion |
| UCMCTrack | `--tracker ucmctrack` | Motion |
| DeepOC-SORT | `--tracker deepocsort` | ReID |
| StrongSORT | `--tracker strongsort` | ReID |
| BoT-SORT | `--tracker botsort` | ReID |
| BoostTrack | `--tracker boosttrack` | ReID |
| HybridSORT | `--tracker hybridsort` | ReID |

## Benchmark Data Structure

```
benchmark_data/
├── MOT17-mini/
│   └── train/
│       ├── MOT17-02-FRCNN/
│       │   ├── det/det.txt
│       │   ├── gt/gt.txt
│       │   ├── img1/
│       │   └── seqinfo.ini
│       └── ...
├── yolox_dets/
│   └── train/
│       ├── MOT17-02.txt
│       └── ...
└── reid_embs/  (optional, for ReID trackers)
    └── train/
        ├── MOT17-02.txt
        └── ...
```

## Detection Format

Input detections file format (one detection per line):
```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
```

For 2D tracking, set `x`, `y`, `z` to `-1`.

Example:
```
1,-1,912,484,97,109,0.95,-1,-1,-1
1,-1,1256,420,85,91,0.87,-1,-1,-1
```

## Output Format

Tracking results follow the MOT Challenge format:
```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
```

## Metrics

| Metric | Description |
|--------|-------------|
| HOTA | Higher Order Tracking Accuracy (primary metric) |
| MOTA | Multiple Object Tracking Accuracy |
| MOTP | Multiple Object Tracking Precision |
| IDF1 | ID F1 Score |
| MT | Mostly Tracked trajectories |
| ML | Mostly Lost trajectories |
| FP | False Positives |
| FN | False Negatives |
| IDSw | ID Switches |

## Creating Custom Benchmark Data

If you want to benchmark on your own dataset:

1. **Organize your data** following the MOT format structure above

2. **Generate detections** using your detector (YOLO, etc.)

3. **Save as text files** in the detection format

4. **Run the benchmark**:
   ```bash
   ./build/tools/motcpp_eval \
       --tracker bytetrack \
       --dataset /path/to/your/dataset \
       --dets custom \
       --output results/
   ```

## Uploading Benchmark Data to Releases

Maintainers can prepare release data:

```bash
./scripts/prepare_release_data.sh
```

Then upload the generated files to GitHub Releases with tag `benchmark-data-v1.0`.

## Expected Results

On MOT17 with YOLOX detections:

| Tracker | HOTA | MOTA | IDF1 |
|---------|------|------|------|
| SORT | ~62 | ~75 | ~69 |
| ByteTrack | ~66 | ~76 | ~77 |
| OC-SORT | ~64 | ~74 | ~74 |
| UCMCTrack | ~64 | ~75 | ~74 |

*Results may vary based on detection quality and parameter tuning.*
