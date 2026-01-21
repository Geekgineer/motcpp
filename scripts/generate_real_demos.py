#!/usr/bin/env python3
"""
Generate REAL demo GIFs by running actual C++ tracker implementations.

This script:
1. Runs motcpp_eval to generate tracking results
2. Visualizes the actual tracking output
3. Creates demo GIFs showing real tracker performance
"""

import os
import sys
import subprocess
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

# Configuration
TRACKERS = ["bytetrack", "ocsort", "boosttrack", "sort"]
SEQUENCE = "MOT17-09"  # Good crowded sequence for demo
MAX_FRAMES = 120
OUTPUT_WIDTH = 640
FPS = 25

# Paths
MOTCPP_ROOT = Path("/media/abi/ext_nvme1/boxmot/motcpp")
MOT_ROOT = Path("/media/abi/ext_nvme1/boxmot/boxmot-cpp/assets/MOT17-ablation/train")
DET_EMB_ROOT = MOTCPP_ROOT / "assets" / "yolox_x_ablation"
EVAL_TOOL = MOTCPP_ROOT / "build" / "tools" / "motcpp_eval"
OUTPUT_DIR = MOTCPP_ROOT / "docs" / "images"
RESULTS_DIR = MOTCPP_ROOT / "demo_results"

def get_color(track_id: int) -> tuple:
    """Generate consistent color for track ID."""
    np.random.seed(track_id * 7 + 13)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))

def run_tracker(tracker_name: str) -> Path:
    """Run the actual C++ tracker and return results path."""
    results_path = RESULTS_DIR / tracker_name
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Run motcpp_eval
    cmd = [
        str(EVAL_TOOL),
        str(MOT_ROOT),
        str(results_path),
        tracker_name,
        str(DET_EMB_ROOT),
        "yolox_x_ablation"
    ]
    
    print(f"  Running: {' '.join(cmd[:4])}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            print(f"  ⚠ Tracker failed: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Tracker timed out")
        return None
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        return None
    
    # Find results file
    results_file = results_path / f"{SEQUENCE}.txt"
    if not results_file.exists():
        # Try without suffix
        for f in results_path.glob("*.txt"):
            if SEQUENCE.replace("-FRCNN", "") in f.name:
                results_file = f
                break
    
    if results_file.exists():
        return results_file
    
    print(f"  ⚠ Results not found")
    return None

def load_tracking_results(results_file: Path) -> dict:
    """Load MOT format tracking results."""
    tracks = defaultdict(list)
    
    with open(results_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                
                tracks[frame].append({
                    'id': track_id,
                    'bbox': [x, y, x + w, y + h],
                    'conf': conf
                })
    
    return tracks

def draw_tracks(frame: np.ndarray, tracks: list, trajectories: dict) -> np.ndarray:
    """Draw tracking boxes and trajectories."""
    overlay = frame.copy()
    
    # Draw trajectories
    for track in tracks:
        track_id = track['id']
        x1, y1, x2, y2 = [int(v) for v in track['bbox']]
        cx, cy = (x1 + x2) // 2, y2
        
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((cx, cy))
        trajectories[track_id] = trajectories[track_id][-30:]
        
        color = get_color(track_id)
        
        points = trajectories[track_id]
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            cv2.line(overlay, points[i-1], points[i], color, thickness)
    
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Draw boxes
    for track in tracks:
        track_id = track['id']
        x1, y1, x2, y2 = [int(v) for v in track['bbox']]
        color = get_color(track_id)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID:{track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def add_overlay(frame: np.ndarray, frame_num: int, num_tracks: int, tracker_name: str) -> np.ndarray:
    """Add info overlay."""
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, "motcpp", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    
    tracker_colors = {
        "bytetrack": (100, 255, 100),
        "ocsort": (255, 150, 100),
        "boosttrack": (255, 100, 255),
        "sort": (100, 200, 255),
    }
    
    display_names = {
        "bytetrack": "ByteTrack",
        "ocsort": "OC-SORT",
        "boosttrack": "BoostTrack",
        "sort": "SORT",
    }
    
    display_name = display_names.get(tracker_name, tracker_name)
    tracker_color = tracker_colors.get(tracker_name, (200, 200, 200))
    cv2.putText(frame, f"| {display_name}", (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracker_color, 2)
    
    # Add "REAL C++" badge
    cv2.putText(frame, "[C++]", (w - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    stats = f"Frame: {frame_num} | Tracks: {num_tracks}"
    (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, stats, (w - tw - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    return frame

def generate_gif(tracker_name: str, results_file: Path) -> bool:
    """Generate GIF from tracking results."""
    # Load tracking results
    tracks_by_frame = load_tracking_results(results_file)
    if not tracks_by_frame:
        print(f"  ⚠ No tracks loaded")
        return False
    
    # Find sequence images
    seq_path = MOT_ROOT / SEQUENCE
    if not seq_path.exists():
        seq_path = MOT_ROOT / SEQUENCE.replace("-FRCNN", "")
    
    img_dir = seq_path / "img1"
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        images = sorted(img_dir.glob("*.png"))
    
    if not images:
        print(f"  ⚠ No images found")
        return False
    
    # Process frames
    frames_out = []
    trajectories = {}
    
    start_frame = 1
    end_frame = min(start_frame + MAX_FRAMES, len(images))
    
    for i, img_path in enumerate(images[start_frame-1:end_frame-1], start=start_frame):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        # Resize
        h, w = frame.shape[:2]
        scale = OUTPUT_WIDTH / w
        new_h = int(h * scale)
        frame = cv2.resize(frame, (OUTPUT_WIDTH, new_h))
        
        # Get tracks for this frame and scale
        tracks = tracks_by_frame.get(i, [])
        scaled_tracks = []
        for t in tracks:
            scaled_tracks.append({
                'id': t['id'],
                'bbox': [v * scale for v in t['bbox']],
                'conf': t['conf']
            })
        
        frame = draw_tracks(frame, scaled_tracks, trajectories)
        frame = add_overlay(frame, i, len(scaled_tracks), tracker_name)
        
        frames_out.append(frame)
    
    if not frames_out:
        print(f"  ⚠ No frames processed")
        return False
    
    # Write video
    display_names = {
        "bytetrack": "bytetrack",
        "ocsort": "ocsort",
        "boosttrack": "boosttrack",
        "sort": "sort",
    }
    
    output_name = f"demo_{display_names.get(tracker_name, tracker_name)}"
    output_video = OUTPUT_DIR / f"{output_name}.mp4"
    output_gif = OUTPUT_DIR / f"{output_name}.gif"
    
    h, w = frames_out[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, FPS, (w, h))
    for f in frames_out:
        out.write(f)
    out.release()
    
    # Convert to GIF
    gif_cmd = f'ffmpeg -y -loglevel error -i "{output_video}" -vf "fps=15,scale={OUTPUT_WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 "{output_gif}"'
    os.system(gif_cmd)
    
    # Cleanup
    output_video.unlink()
    
    size_mb = output_gif.stat().st_size / (1024 * 1024)
    print(f"  ✓ {output_gif.name} ({size_mb:.1f}MB)")
    
    return True

def main():
    print("=== Generating REAL Demo GIFs with C++ Trackers ===\n")
    
    # Check prerequisites
    if not EVAL_TOOL.exists():
        print(f"Error: motcpp_eval not found at {EVAL_TOOL}")
        print("Build with: cmake --build build --target motcpp_eval")
        sys.exit(1)
    
    if not MOT_ROOT.exists():
        print(f"Error: MOT data not found at {MOT_ROOT}")
        sys.exit(1)
    
    if not DET_EMB_ROOT.exists():
        print(f"Error: Detections not found at {DET_EMB_ROOT}")
        sys.exit(1)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for tracker in TRACKERS:
        print(f"\n[{tracker.upper()}]")
        
        # Run actual tracker
        print(f"  Running C++ tracker...")
        results_file = run_tracker(tracker)
        
        if results_file is None:
            print(f"  ⚠ Skipping {tracker}")
            continue
        
        print(f"  Results: {results_file}")
        
        # Generate GIF
        print(f"  Generating GIF...")
        if generate_gif(tracker, results_file):
            success_count += 1
    
    print(f"\n=== Done! Generated {success_count}/{len(TRACKERS)} REAL GIFs ===")
    print(f"Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
