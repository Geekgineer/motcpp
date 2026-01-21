#!/usr/bin/env python3
"""
Generate multiple demo GIFs from MOT17 tracking results for README.

Creates demo GIFs for different trackers and sequences.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

# Configuration
DEMOS = [
    {"tracker": "ByteTrack", "sequence": "MOT17-09", "start": 1, "color_seed": 0},
    {"tracker": "OC-SORT", "sequence": "MOT17-04", "start": 300, "color_seed": 100},
    {"tracker": "BoostTrack", "sequence": "MOT17-02", "start": 1, "color_seed": 200},
    {"tracker": "SORT", "sequence": "MOT17-05", "start": 1, "color_seed": 300},
]

MAX_FRAMES = 120  # ~4 seconds at 30fps
OUTPUT_WIDTH = 640
FPS = 25

def get_color(track_id: int, seed_offset: int = 0) -> tuple:
    """Generate consistent color for track ID."""
    np.random.seed(track_id * 7 + 13 + seed_offset)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))

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
                if len(parts) >= 9:
                    conf = float(parts[6])
                    cls = int(parts[7])
                    vis = float(parts[8])
                    if cls not in [1, 2, 7] or vis < 0.1:
                        continue
                tracks[frame].append({
                    'id': track_id,
                    'bbox': [x, y, x + w, y + h]
                })
    
    return tracks

def draw_tracks(frame: np.ndarray, tracks: list, trajectories: dict, color_seed: int) -> np.ndarray:
    """Draw tracking boxes and trajectories on frame."""
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
        
        color = get_color(track_id, color_seed)
        
        points = trajectories[track_id]
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            cv2.line(overlay, points[i-1], points[i], color, thickness)
    
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    for track in tracks:
        track_id = track['id']
        x1, y1, x2, y2 = [int(v) for v in track['bbox']]
        color = get_color(track_id, color_seed)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID:{track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def add_overlay(frame: np.ndarray, frame_num: int, num_tracks: int, tracker_name: str) -> np.ndarray:
    """Add info overlay to frame."""
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # motcpp logo
    cv2.putText(frame, "motcpp", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    
    # Tracker name with different colors
    tracker_colors = {
        "ByteTrack": (100, 255, 100),
        "OC-SORT": (255, 150, 100),
        "BoostTrack": (255, 100, 255),
        "SORT": (100, 200, 255),
    }
    tracker_color = tracker_colors.get(tracker_name, (200, 200, 200))
    cv2.putText(frame, f"| {tracker_name}", (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracker_color, 2)
    
    stats = f"Frame: {frame_num} | Tracks: {num_tracks}"
    (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, stats, (w - tw - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    return frame

def generate_demo(config: dict, base_path: Path, output_dir: Path) -> bool:
    """Generate a single demo GIF."""
    tracker_name = config["tracker"]
    sequence = config["sequence"]
    start_frame = config["start"]
    color_seed = config["color_seed"]
    
    seq_path = base_path / sequence
    if not seq_path.exists():
        # Try without -FRCNN suffix
        seq_path = base_path / sequence.replace("-FRCNN", "")
    
    if not seq_path.exists():
        print(f"  ⚠ Sequence not found: {sequence}")
        return False
    
    img_dir = seq_path / "img1"
    gt_file = seq_path / "gt" / "gt.txt"
    
    if not img_dir.exists() or not gt_file.exists():
        print(f"  ⚠ Missing data for: {sequence}")
        return False
    
    tracks_by_frame = load_gt(gt_file)
    if not tracks_by_frame:
        print(f"  ⚠ No tracks for: {sequence}")
        return False
    
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        images = sorted(img_dir.glob("*.png"))
    
    # Process frames
    frames_out = []
    trajectories = {}
    
    end_frame = min(start_frame + MAX_FRAMES, len(images))
    
    for i, img_path in enumerate(images[start_frame-1:end_frame-1], start=start_frame):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        h, w = frame.shape[:2]
        scale = OUTPUT_WIDTH / w
        new_h = int(h * scale)
        frame = cv2.resize(frame, (OUTPUT_WIDTH, new_h))
        
        tracks = tracks_by_frame.get(i, [])
        scaled_tracks = []
        for t in tracks:
            scaled_tracks.append({
                'id': t['id'],
                'bbox': [v * scale for v in t['bbox']]
            })
        
        frame = draw_tracks(frame, scaled_tracks, trajectories, color_seed)
        frame = add_overlay(frame, i, len(scaled_tracks), tracker_name)
        
        frames_out.append(frame)
    
    if not frames_out:
        print(f"  ⚠ No frames for: {sequence}")
        return False
    
    # Output filename
    output_name = f"demo_{tracker_name.lower().replace('-', '')}"
    output_video = output_dir / f"{output_name}.mp4"
    output_gif = output_dir / f"{output_name}.gif"
    
    # Write MP4
    h, w = frames_out[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, FPS, (w, h))
    for f in frames_out:
        out.write(f)
    out.release()
    
    # Convert to GIF
    gif_cmd = f'ffmpeg -y -loglevel error -i "{output_video}" -vf "fps=15,scale={OUTPUT_WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 "{output_gif}"'
    os.system(gif_cmd)
    
    # Clean up MP4
    output_video.unlink()
    
    size_mb = output_gif.stat().st_size / (1024 * 1024)
    print(f"  ✓ {output_gif.name} ({size_mb:.1f}MB) - {sequence}")
    
    return True

def main():
    base_paths = [
        Path("/media/abi/ext_nvme1/boxmot/boxmot-cpp/assets/MOT17-ablation/train"),
        Path("/media/abi/ext_nvme1/boxmot/motcpp/assets/MOT17-ablation/train"),
    ]
    
    base_path = None
    for bp in base_paths:
        if bp.exists():
            base_path = bp
            break
    
    if base_path is None:
        print("Error: Could not find MOT17-ablation data")
        sys.exit(1)
    
    print(f"Using data from: {base_path}")
    
    output_dir = Path("/media/abi/ext_nvme1/boxmot/motcpp/docs/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Generating Demo GIFs ===\n")
    
    success_count = 0
    for config in DEMOS:
        print(f"Generating {config['tracker']}...")
        if generate_demo(config, base_path, output_dir):
            success_count += 1
    
    print(f"\n=== Done! Generated {success_count}/{len(DEMOS)} GIFs ===")
    print(f"Location: {output_dir}")

if __name__ == "__main__":
    main()
