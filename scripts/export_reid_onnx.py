#!/usr/bin/env python3
"""
Standalone script to export ReID models to ONNX format for BoxMOT C++.
This script uses the boxmot Python package from the parent directory.
"""
import sys
import os
from pathlib import Path

# Add parent boxmot directory to path
script_dir = Path(__file__).parent
boxmot_python_dir = script_dir.parent.parent / "boxmot"
sys.path.insert(0, str(boxmot_python_dir))

def export_model(model_name="osnet_x1_0_dukemtmcreid", output_dir=None):
    """Export a ReID model to ONNX format."""
    if output_dir is None:
        output_dir = script_dir / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import after path setup
    try:
        from appearance.reid.auto_backend import ReidAutoBackend
        from appearance.exporters.onnx_exporter import ONNXExporter
        import torch
    except ImportError as e:
        print(f"Error importing boxmot modules: {e}")
        print(f"Make sure boxmot is available at: {boxmot_python_dir}")
        print("\nTrying to install dependencies...")
        import subprocess
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "onnx", "onnxruntime", "numpy", "pandas", "-q"], check=True)
            print("Dependencies installed. Please run the script again.")
        except:
            print("Failed to install dependencies automatically.")
            print("Please install manually: pip install torch onnx onnxruntime numpy pandas")
        return None
    
    # Setup model
    device = torch.device("cpu")
    print(f"Loading model: {model_name}")
    
    try:
        auto_backend = ReidAutoBackend(weights=model_name, device=device, half=False)
        model = auto_backend.model.model.eval()
        
        # Create dummy input (batch_size=1, channels=3, height=256, width=128)
        dummy_input = torch.empty(1, 3, 256, 128).to(device)
        
        # Warmup
        for _ in range(2):
            _ = model(dummy_input)
        
        # Export to ONNX
        output_file = output_dir / f"{model_name}.onnx"
        print(f"Exporting to {output_file}...")
        
        exporter = ONNXExporter(
            model=model,
            im=dummy_input,
            file=output_file,
            optimize=False,
            dynamic=True,  # Enable dynamic batch size
            half=False,
            simplify=False
        )
        
        exported_file = exporter.export()
        
        if exported_file and Path(exported_file).exists():
            file_size = Path(exported_file).stat().st_size / (1024 * 1024)
            print(f"✅ Successfully exported {exported_file} ({file_size:.1f} MB)")
            return exported_file
        else:
            print("❌ Export failed")
            return None
            
    except Exception as e:
        print(f"❌ Error during export: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Export ReID models to ONNX format")
    parser.add_argument("--model", type=str, default="osnet_x1_0_dukemtmcreid",
                       help="Model name (e.g., osnet_x1_0_dukemtmcreid)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for ONNX models (default: scripts/models)")
    
    args = parser.parse_args()
    
    # Export model
    output_file = export_model(args.model, args.output_dir)
    
    if output_file:
        print(f"\n✅ Model exported successfully!")
        print(f"   File: {output_file}")
        print(f"\nYou can now use this model with DeepOCSort:")
        print(f"   cd build/tools")
        print(f"   ./boxmot_eval <mot_root> <output_dir> deepocsort \"\" \"\" \"\" {output_file}")
    else:
        print("\n❌ Model export failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
