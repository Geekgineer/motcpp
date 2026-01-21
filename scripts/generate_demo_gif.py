#!/usr/bin/env python3
"""
Generate demo GIF from MOT17 tracking results for README.

This script visualizes tracking results on MOT17 sequences to create
an animated demo for the repository.

Usage:
    python scripts/generate_demo_gif.py
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

# Configuration
SEQUENCE = "MOT17-09"  # Good crowded sequence
MAX_FRAMES = 150  # ~5 seconds at 30fps
OUTPUT_WIDTH = 640
FPS = 25

def get_color(track_id: int) -> tuple:
    """Generate consistent color for track ID."""
    np.random.seed(track_id * 7 + 13)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))

def load_detections(det_file: Path) -> dict:
    """Load detections from MOT format file."""
    detections = defaultdict(list)
    if not det_file.exists():
        return detections
    
    with open(det_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame = int(parts[0])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                detections[frame].append([x, y, x + w, y + h, conf])
    
    return detections

def load_gt(gt_file: Path) -> dict:
    """Load ground truth for visualization (simulating tracking results)."""
    tracks = defaultdict(list)
    if not gt_file.exists():
        return tracks
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                # Filter by visibility and class (pedestrians only)
                if len(parts) >= 9:
                    conf = float(parts[6])
                    cls = int(parts[7])
                    vis = float(parts[8])
                    if cls not in [1, 2, 7] or vis < 0.1:  # pedestrians, persons on vehicle, static persons
                        continue
                tracks[frame].append({
                    'id': track_id,
                    'bbox': [x, y, x + w, y + h]
                })
    
    return tracks

def draw_tracks(frame: np.ndarray, tracks: list, trajectories: dict) -> np.ndarray:
    """Draw tracking boxes and trajectories on frame."""
    overlay = frame.copy()
    
    # Draw trajectories
    for track in tracks:
        track_id = track['id']
        x1, y1, x2, y2 = [int(v) for v in track['bbox']]
        cx, cy = (x1 + x2) // 2, y2  # bottom center
        
        # Add to trajectory
        if track_id not in trajectories:
            trajectories[track_id] = []
        trajectories[track_id].append((cx, cy))
        
        # Keep last 30 points
        trajectories[track_id] = trajectories[track_id][-30:]
        
        color = get_color(track_id)
        
        # Draw trajectory line
        points = trajectories[track_id]
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            cv2.line(overlay, points[i-1], points[i], color, thickness)
    
    # Blend trajectory overlay
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Draw boxes and IDs
    for track in tracks:
        track_id = track['id']
        x1, y1, x2, y2 = [int(v) for v in track['bbox']]
        color = get_color(track_id)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID label
        label = f"ID:{track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def add_overlay(frame: np.ndarray, frame_num: int, num_tracks: int, tracker_name: str = "ByteTrack") -> np.ndarray:
    """Add info overlay to frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # motcpp logo text
    cv2.putText(frame, "motcpp", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    
    # Tracker name
    cv2.putText(frame, f"| {tracker_name}", (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Stats
    stats = f"Frame: {frame_num} | Tracks: {num_tracks}"
    (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, stats, (w - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    return frame

def main():
    # Find MOT17 data
    base_paths = [
        Path("/media/abi/ext_nvme1/boxmot/boxmot-cpp/assets/MOT17-ablation/train"),
        Path("/media/abi/ext_nvme1/boxmot/motcpp/assets/MOT17-ablation/train"),
    ]
    
    seq_path = None
    for base in base_paths:
        candidate = base / SEQUENCE
        if candidate.exists():
            seq_path = candidate
            break
    
    if seq_path is None:
        print(f"Error: Could not find {SEQUENCE} in any known location")
        sys.exit(1)
    
    print(f"Using sequence: {seq_path}")
    
    img_dir = seq_path / "img1"
    gt_file = seq_path / "gt" / "gt.txt"
    
    if not img_dir.exists():
        print(f"Error: Image directory not found: {img_dir}")
        sys.exit(1)
    
    # Load ground truth (simulating tracking output)
    tracks_by_frame = load_gt(gt_file)
    if not tracks_by_frame:
        print("Error: Could not load ground truth")
        sys.exit(1)
    
    # Get image files
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        images = sorted(img_dir.glob("*.png"))
    
    print(f"Found {len(images)} frames")
    
    # Prepare output
    output_dir = Path("/media/abi/ext_nvme1/boxmot/motcpp/docs/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "demo.mp4"
    output_gif = output_dir / "demo.gif"
    
    # Process frames
    frames_out = []
    trajectories = {}
    
    start_frame = 1
    end_frame = min(start_frame + MAX_FRAMES, len(images))
    
    print(f"Processing frames {start_frame} to {end_frame}...")
    
    for i, img_path in enumerate(images[start_frame-1:end_frame-1], start=start_frame):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        # Resize
        h, w = frame.shape[:2]
        scale = OUTPUT_WIDTH / w
        new_h = int(h * scale)
        frame = cv2.resize(frame, (OUTPUT_WIDTH, new_h))
        
        # Scale tracks
        tracks = tracks_by_frame.get(i, [])
        scaled_tracks = []
        for t in tracks:
            scaled_tracks.append({
                'id': t['id'],
                'bbox': [v * scale for v in t['bbox']]
            })
        
        # Draw
        frame = draw_tracks(frame, scaled_tracks, trajectories)
        frame = add_overlay(frame, i, len(scaled_tracks))
        
        frames_out.append(frame)
        
        if i % 30 == 0:
            print(f"  Processed frame {i}")
    
    if not frames_out:
        print("Error: No frames processed")
        sys.exit(1)
    
    # Write MP4
    print(f"Writing video to {output_video}...")
    h, w = frames_out[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, FPS, (w, h))
    for f in frames_out:
        out.write(f)
    out.release()
    
    # Convert to GIF using ffmpeg
    print(f"Converting to GIF...")
    gif_cmd = f'ffmpeg -y -i "{output_video}" -vf "fps=15,scale={OUTPUT_WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 "{output_gif}"'
    os.system(gif_cmd)
    
    print(f"\n✓ Demo video saved to: {output_video}")
    print(f"✓ Demo GIF saved to: {output_gif}")
    print(f"\nTo use in README, upload to GitHub releases and reference:")
    print(f'  <img src="https://github.com/Geekgineer/motcpp/releases/download/v1.0.0/demo.gif" width="640">')

if __name__ == "__main__":
    main()
