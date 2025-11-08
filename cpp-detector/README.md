# DetectCodeGPT C++ Edition

An enhanced version of DetectCodeGPT specifically designed to detect AI-generated C++ code across multiple projects. Features GPU acceleration, a user-friendly GUI, and comprehensive JSON reporting.

## Features

- **Multi-Project Support**: Scan entire directories containing multiple C++ projects
- **GPU Acceleration**: Automatically detects and uses available GPUs (CUDA, MPS) for faster processing
- **User-Friendly GUI**: Intuitive interface for configuration and real-time progress monitoring
- **CLI Mode**: Command-line interface for automation and scripting
- **Comprehensive Output**: JSON results with detailed scores and human-readable reports
- **Flexible Configuration**: Customizable detection parameters and model selection

## Installation

### Prerequisites

- Python 3.9.7 or higher
- CUDA-capable GPU (optional, but recommended for better performance)

### Install Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

Or install specific requirements for the C++ detector:

```bash
cd cpp-detector
pip install -r requirements.txt
```

## Usage

### GUI Mode (Recommended)

Launch the graphical interface:

```bash
cd cpp-detector
python main_app.py
```

The GUI provides:
- **Configuration Tab**: Set up directories, models, and detection parameters
- **Processing Tab**: Monitor real-time progress and logs
- **Results Tab**: View detection results and export reports

### CLI Mode

For automation or headless environments:

```bash
# Basic usage
python main_app.py --cli --directory /path/to/cpp/projects

# With custom settings
python main_app.py --cli \
    --directory /path/to/cpp/projects \
    --output results.json \
    --report report.txt \
    --n-perturbations 100 \
    --device cuda

# Show available devices
python main_app.py --device-info
```

### CLI Options

```
--cli                    Run in CLI mode instead of GUI
--directory, -d          Directory to scan for C++ files (required for CLI)
--output, -o             Output JSON file path
--report, -r             Generate text report at specified path
--base-model             Base model for scoring (default: codellama/CodeLlama-7b-hf)
--mask-model             Mask filling model (default: Salesforce/codet5p-770m)
--device                 Device to use: cuda, cpu, or mps (default: auto-detect)
--n-perturbations        Number of perturbations per sample (default: 50)
--batch-size             Batch size for processing (default: 10)
--min-size               Minimum file size in bytes (default: 100)
--max-size               Maximum file size in bytes (default: 100000)
--max-files              Maximum files to process, 0 for all (default: 0)
--device-info            Show device information and exit
--debug                  Enable debug mode with full tracebacks
```

## How It Works

DetectCodeGPT C++ Edition uses a multi-step detection process:

1. **File Scanning**: Recursively scans directories for C++ files (.cpp, .hpp, .h, .cc, etc.)
2. **Project Organization**: Groups files by project based on build system files (CMakeLists.txt, Makefile, etc.)
3. **Code Preprocessing**: Normalizes and prepares C++ code for analysis
4. **Perturbation Generation**: Creates multiple perturbed versions of each code sample
5. **Likelihood Scoring**: Computes log-likelihood and rank scores using language models
6. **Detection**: Applies DetectCodeGPT algorithm to classify code as AI-generated or human-written

### Detection Scores

The tool provides three main detection scores:

- **DetectCodeGPT Score (NPR)**: Normalized Perturbation Rank - primary metric
  - Values > 1.0: Likely AI-generated
  - Values 0.95-1.0: Possibly AI-generated
  - Values < 0.95: Likely human-written

- **DetectGPT Score**: Curvature-based detection from original DetectGPT paper
- **LRR Score**: Log-likelihood to Rank Ratio

## Output Format

### JSON Output

The tool generates a comprehensive JSON file with:

```json
{
  "metadata": {
    "root_path": "/path/to/projects",
    "timestamp": "2025-11-08T12:00:00",
    "processing_time_seconds": 123.45,
    "files_processed": 50,
    "device_info": {...}
  },
  "summary": {
    "total_analyzed": 50,
    "likely_ai_generated": 5,
    "possibly_ai_generated": 3,
    "likely_human_written": 42,
    "ai_percentage": 10.0
  },
  "projects": {
    "project_name": [
      {
        "file_path": "/full/path/to/file.cpp",
        "relative_path": "src/file.cpp",
        "detection": {
          "detectcodegpt_score": 1.05,
          "prediction": "likely_ai_generated",
          "confidence": "high",
          ...
        }
      }
    ]
  }
}
```

### Text Report

Generate a human-readable summary:

```
================================================================================
C++ AI-Generated Code Detection Report
================================================================================

Root Path: /path/to/projects
Timestamp: 2025-11-08T12:00:00
Processing Time: 123.45 seconds
Device: cuda

Summary:
  Total Files Analyzed: 50
  Likely AI-Generated: 5 (10.0%)
  Possibly AI-Generated: 3
  Likely Human-Written: 42
  Errors: 0

Projects:
  project1:
    Files: 25
    Likely AI-Generated: 2
    Suspicious files:
      - src/algorithm.cpp (score: 1.0234)
      - src/utils.cpp (score: 1.0156)
```

## Supported C++ File Types

- `.cpp` - C++ source files
- `.cc` - C++ source files (alternative extension)
- `.cxx` - C++ source files (alternative extension)
- `.c++` - C++ source files (alternative extension)
- `.hpp` - C++ header files
- `.h` - C/C++ header files
- `.hh` - C++ header files (alternative extension)
- `.hxx` - C++ header files (alternative extension)
- `.h++` - C++ header files (alternative extension)

## GPU Support

The tool automatically detects and uses available GPUs:

- **NVIDIA GPUs**: CUDA support (recommended)
- **Apple Silicon**: Metal Performance Shaders (MPS)
- **CPU Fallback**: Works without GPU but slower

Check GPU availability:

```bash
python main_app.py --device-info
```

## Model Options

### Base Models (for scoring)

- `codellama/CodeLlama-7b-hf` (default, recommended for C++)
- `codellama/CodeLlama-13b-hf` (larger, more accurate, slower)
- `Salesforce/codegen-2B-mono`
- `microsoft/CodeGPT-small-py`

### Mask Filling Models

- `Salesforce/codet5p-770m` (default, recommended)
- `Salesforce/codet5-base`
- `Salesforce/CodeT5-large`

## Performance Tips

1. **Use GPU**: Significant speedup with CUDA-capable GPU
2. **Adjust Batch Size**: Increase for faster processing (if memory allows)
3. **Reduce Perturbations**: Lower `n-perturbations` for faster but less accurate results
4. **File Size Limits**: Adjust `min-size` and `max-size` to focus on relevant files
5. **File Limit**: Use `max-files` for quick testing on large codebases

## Troubleshooting

### Out of Memory Errors

- Reduce `batch-size`
- Reduce `n-perturbations`
- Use a smaller base model
- Process fewer files at once with `max-files`

### Slow Processing

- Ensure GPU is being used (check with `--device-info`)
- Increase `batch-size` if memory allows
- Reduce `n-perturbations`

### Import Errors

Make sure all dependencies are installed:

```bash
pip install -r ../requirements.txt
```

## Examples

### Scan a single project

```bash
python main_app.py --cli --directory ~/my_cpp_project --output results.json
```

### Scan with detailed reporting

```bash
python main_app.py --cli \
    --directory ~/cpp_projects \
    --output results.json \
    --report detailed_report.txt \
    --n-perturbations 100
```

### Quick scan for testing

```bash
python main_app.py --cli \
    --directory ~/large_codebase \
    --max-files 10 \
    --n-perturbations 25
```

## License

This project extends DetectCodeGPT and is licensed under the MIT License. See the LICENSE file in the repository root for details.

## Citation

If you use this tool in your research, please cite the original DetectCodeGPT paper:

```bibtex
@inproceedings{shi2025detectcodegpt,
  title={Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers},
  author={Shi, Yuling and Zhang, Hongyu and Wan, Chengcheng and Gu, Xiaodong},
  booktitle={Proceedings of the 47th International Conference on Software Engineering (ICSE 2025)},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgements

Based on DetectCodeGPT by Shi et al., with enhancements for C++ code analysis, GPU acceleration, and user interface improvements.
