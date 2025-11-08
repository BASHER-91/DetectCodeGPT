"""
Script to check GPU availability and PyTorch CUDA configuration.
"""
import torch
import sys

print("=" * 60)
print("GPU/CUDA Configuration Check")
print("=" * 60)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("\n❌ No CUDA support detected!")
    print("\nPossible reasons:")
    print("1. No NVIDIA GPU installed")
    print("2. GPU drivers not installed or outdated")
    print("3. PyTorch installed without CUDA support (CPU-only version)")
    print("\nTo install PyTorch with CUDA support:")
    print("Visit: https://pytorch.org/get-started/locally/")
    print("\nFor CUDA 11.8:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\nFor CUDA 12.1:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 60)
