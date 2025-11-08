"""
DetectCodeGPT C++ Edition
AI-generated C++ code detection with GPU acceleration and GUI.
"""

__version__ = "1.0.0"
__author__ = "DetectCodeGPT Contributors"

from .cpp_detector import CppCodeDetector, GPUManager
from .file_scanner import CppFileScanner, scan_cpp_files
from .batch_processor import BatchProcessor

__all__ = [
    'CppCodeDetector',
    'GPUManager',
    'CppFileScanner',
    'scan_cpp_files',
    'BatchProcessor'
]
