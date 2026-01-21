#!/usr/bin/env python3
"""
Simple MOT evaluation script for BoxMOT C++ results
Computes basic metrics: HOTA, MOTA, IDF1 using TrackEval
"""

import sys
import os
import argparse
from pathlib import Path
import subprocess
import tempfile
import shutil
import zipfile
import urllib.request

def find_trackeval_scripts():
    """Find TrackEval scripts - either from downloaded repo or use uv run."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Check if TrackEval is downloaded in project
    trackeval_dir = project_root / "trackeval"
    trackeval_script = trackeval_dir / "scripts" / "run_mot_challenge.py"
    if trackeval_script.exists():
        return trackeval_script
    
    # Try to find .venv Python
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        try:
            result = subprocess.run(
                [str(venv_python), "-c", "import trackeval"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                # Download TrackEval scripts
                print("Downloading TrackEval scripts...")
                download_trackeval_scripts(trackeval_dir)
                if trackeval_script.exists():
                    return trackeval_script
        except:
            pass
    
    return None

def patch_trackeval_numpy(root: Path):
    """Patch TrackEval to fix numpy 2.0 compatibility issues."""
    import re
    
    # Files that might have np.float, np.int, np.bool
    deprecated_types = {
        'float': 'float64',
        'int': 'int64', 
        'bool': 'bool_',
        'complex': 'complex128'
    }
    
    for file_path in root.rglob("*.py"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            for deprecated, replacement in deprecated_types.items():
                # Replace np.float, np.int, etc. with np.float64, np.int64, etc.
                pattern = rf'\bnp\.{deprecated}\b'
                content = re.sub(pattern, f'np.{replacement}', content)
            
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        except Exception:
            pass  # Skip files we can't read/write

def download_trackeval_scripts(dest: Path):
    """Download TrackEval repository scripts."""
    if dest.exists():
        patch_trackeval_numpy(dest)
        return
    
    print(f"Downloading TrackEval to {dest}...")
    repo_url = "https://github.com/JonathonLuiten/TrackEval/archive/refs/heads/master.zip"
    zip_file = dest.parent / "TrackEval-master.zip"
    
    try:
        urllib.request.urlretrieve(repo_url, zip_file)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dest.parent)
        
        # Rename extracted folder
        extracted = dest.parent / "TrackEval-master"
        if extracted.exists():
            extracted.rename(dest)
        
        zip_file.unlink()
        
        # Patch for numpy 2.0 compatibility
        patch_trackeval_numpy(dest)
        
        print(f"✓ TrackEval downloaded and patched to {dest}")
    except Exception as e:
        print(f"Warning: Could not download TrackEval: {e}")
        if zip_file.exists():
            zip_file.unlink()

def main():
    parser = argparse.ArgumentParser(description='Evaluate MOT tracking results')
    parser.add_argument('--gt_folder', type=str, required=True,
                       help='Path to ground truth folder (e.g., MOT17-mini/train)')
    parser.add_argument('--trackers_folder', type=str, required=True,
                       help='Path to tracker results folder')
    parser.add_argument('--tracker_name', type=str, default='bytetrack',
                       help='Name of the tracker (default: bytetrack)')
    parser.add_argument('--no_trackeval', action='store_true',
                       help='Skip TrackEval and only do basic validation')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip basic validation and go straight to TrackEval')
    
    args = parser.parse_args()
    
    # Find TrackEval scripts
    trackeval_script = find_trackeval_scripts()
    if trackeval_script is None:
        print("Error: Could not find TrackEval scripts.")
        print("Please install trackeval with: uv pip install trackeval")
        print("Or download TrackEval repository manually.")
        return 1
    
    # Determine Python executable
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        python_exe = str(venv_python)
    else:
        python_exe = sys.executable
    
    gt_folder = Path(args.gt_folder).resolve()
    trackers_folder = Path(args.trackers_folder).resolve()
    
    if not gt_folder.exists():
        print(f"Error: GT folder not found: {gt_folder}")
        return 1
    
    if not trackers_folder.exists():
        print(f"Error: Trackers folder not found: {trackers_folder}")
        return 1
    
    print("BoxMOT C++ Evaluation")
    print("=" * 50)
    print(f"GT Folder: {gt_folder}")
    print(f"Results Folder: {trackers_folder}\n")
    
    # Basic validation: check result files exist for each sequence
    seq_dirs = [d for d in gt_folder.iterdir() if d.is_dir() and (d / "img1").exists()]
    
    if not args.skip_validation:
        print(f"Found {len(seq_dirs)} sequences")
        print("\nResults validation:")
        print("-" * 50)
        
        all_valid = True
        for seq_dir in sorted(seq_dirs):
            seq_name = seq_dir.name
            result_file = trackers_folder / f"{seq_name}.txt"
            
            if result_file.exists():
                with open(result_file) as f:
                    lines = [l for l in f if l.strip() and not l.startswith('#')]
                print(f"✓ {seq_name}: {len(lines)} tracked objects")
            else:
                print(f"✗ {seq_name}: Result file not found")
                all_valid = False
        
        if not all_valid:
            print("\n✗ Some sequences are missing results")
            return 1
    
    # Use TrackEval for evaluation
    if not args.no_trackeval:
        try:
            # TrackEval expects a specific structure:
            # TRACKERS_FOLDER/
            #   tracker_name/
            #     data/
            #       sequence1.txt
            #       sequence2.txt
            
            # Create temporary directory structure for TrackEval
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                tracker_dir = tmp_path / args.tracker_name / "data"
                tracker_dir.mkdir(parents=True)
                
                # Copy result files to the expected structure
                print(f"\nPreparing TrackEval directory structure...")
                for seq_dir in sorted(seq_dirs):
                    seq_name = seq_dir.name
                    result_file = trackers_folder / f"{seq_name}.txt"
                    if result_file.exists():
                        shutil.copy2(result_file, tracker_dir / f"{seq_name}.txt")
                
                # Determine split from GT folder path (train/test)
                split = "train"  # default
                if "test" in str(gt_folder).lower():
                    split = "test"
                elif gt_folder.name == "test":
                    split = "test"
                
                # Get sequence names
                seq_names = [seq_dir.name for seq_dir in sorted(seq_dirs)]
                
                # Run TrackEval (similar to boxmot's approach)
                print(f"Running TrackEval evaluation...\n")
                print(f"Using Python: {python_exe}\n")
                
                # Build command arguments
                cmd_args = [
                    "--GT_FOLDER", str(gt_folder),
                    "--BENCHMARK", "",
                    "--TRACKERS_FOLDER", str(tmp_path),
                    "--TRACKERS_TO_EVAL", args.tracker_name,
                    "--SPLIT_TO_EVAL", split,
                    "--METRICS", "HOTA", "CLEAR", "Identity",
                    "--USE_PARALLEL", "False",
                    "--TRACKER_SUB_FOLDER", "data",
                    "--SKIP_SPLIT_FOL", "True",
                    "--GT_LOC_FORMAT", "{gt_folder}/{seq}/gt/gt.txt",
                    "--SEQ_INFO", *seq_names,
                ]
                
                # Build command - trackeval_script should be a Path at this point
                if not isinstance(trackeval_script, Path):
                    print("Error: TrackEval script not found. The script should have downloaded it.")
                    return 1
                
                cmd = [python_exe, str(trackeval_script)] + cmd_args
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(result.stdout)
                    print("\n✓ Evaluation completed successfully!")
                else:
                    print("Error running TrackEval:")
                    print(result.stderr)
                    if result.stdout:
                        print(result.stdout)
                    return 1
                    
        except ImportError:
            print("Error: TrackEval not available.")
            print("Install with: uv pip install trackeval")
            return 1
        except Exception as e:
            print(f"Error running TrackEval: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n✓ All sequences processed successfully!")
        print(f"\nResults are ready for evaluation.")
        print(f"To compute detailed metrics with TrackEval:")
        print(f"  python3 {sys.argv[0]} --gt_folder {gt_folder} --trackers_folder {trackers_folder}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

