#!/usr/bin/env python3
"""
DetectCodeGPT C++ Edition - Main Application Entry Point
Supports both GUI and CLI modes for detecting AI-generated C++ code.
"""

import sys
import argparse
from pathlib import Path
from loguru import logger

# Configure logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


def run_gui():
    """Run the GUI application."""
    try:
        from gui import main as gui_main
        logger.info("Starting GUI application...")
        gui_main()
    except ImportError as e:
        logger.error(f"Failed to import GUI module: {e}")
        logger.error("Make sure tkinter is installed: pip install tk")
        sys.exit(1)


def run_cli(args):
    """
    Run the CLI application.

    Args:
        args: Parsed command-line arguments
    """
    from batch_processor import BatchProcessor

    if not args.directory:
        logger.error("Directory path is required for CLI mode")
        sys.exit(1)

    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory does not exist: {args.directory}")
        sys.exit(1)

    # Create output path if not specified
    if not args.output:
        timestamp = Path(directory).name
        args.output = f"results_{timestamp}.json"

    logger.info("=" * 80)
    logger.info("DetectCodeGPT C++ Edition - CLI Mode")
    logger.info("=" * 80)
    logger.info(f"Directory: {args.directory}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"Mask Model: {args.mask_model}")
    logger.info(f"Device: {args.device if args.device else 'auto'}")
    logger.info(f"Perturbations: {args.n_perturbations}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info("=" * 80)

    # Create processor
    processor = BatchProcessor(
        base_model_name=args.base_model,
        mask_filling_model_name=args.mask_model,
        device=args.device,
        batch_size=args.batch_size,
        n_perturbations=args.n_perturbations
    )

    # Process directory
    try:
        max_files = args.max_files if args.max_files > 0 else None

        results = processor.process_directory(
            root_path=args.directory,
            output_path=args.output,
            min_size=args.min_size,
            max_size=args.max_size,
            max_files=max_files
        )

        # Display summary
        logger.info("\n" + "=" * 80)
        logger.info("DETECTION COMPLETE")
        logger.info("=" * 80)

        summary = results.get('summary', {})
        logger.info(f"Total Files Analyzed: {summary.get('total_analyzed', 0)}")
        logger.info(f"Likely AI-Generated: {summary.get('likely_ai_generated', 0)}")
        logger.info(f"Possibly AI-Generated: {summary.get('possibly_ai_generated', 0)}")
        logger.info(f"Likely Human-Written: {summary.get('likely_human_written', 0)}")
        logger.info(f"AI Percentage: {summary.get('ai_percentage', 0)}%")
        logger.info(f"\nResults saved to: {args.output}")

        # Generate text report if requested
        if args.report:
            report = processor.generate_report(results, output_path=args.report)
            logger.info(f"Report saved to: {args.report}")

        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        if args.debug:
            raise
        sys.exit(1)


def show_device_info():
    """Display information about available devices."""
    from cpp_detector import GPUManager

    logger.info("=" * 80)
    logger.info("DEVICE INFORMATION")
    logger.info("=" * 80)

    device_info = GPUManager.get_device_info()

    logger.info(f"CUDA Available: {device_info['cuda_available']}")
    logger.info(f"MPS Available: {device_info['mps_available']}")
    logger.info(f"Device Count: {device_info['device_count']}")

    if device_info['devices']:
        logger.info("\nAvailable GPUs:")
        for device in device_info['devices']:
            logger.info(f"  GPU {device['id']}: {device['name']}")
            logger.info(f"    Total Memory: {device['total_memory'] / (1024**3):.2f} GB")
            logger.info(f"    Compute Capability: {device['capability']}")
    else:
        logger.info("\nNo GPUs available. Will use CPU for computation.")

    logger.info(f"\nRecommended Device: {GPUManager.get_device()}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DetectCodeGPT C++ Edition - Detect AI-generated C++ code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI
  python main_app.py

  # CLI mode - scan directory
  python main_app.py --cli --directory /path/to/cpp/project

  # CLI mode with custom settings
  python main_app.py --cli --directory /path/to/cpp/project \\
      --output results.json --n-perturbations 100 --device cuda

  # Show device information
  python main_app.py --device-info
        """
    )

    parser.add_argument('--cli', action='store_true',
                        help='Run in CLI mode instead of GUI')

    parser.add_argument('--directory', '-d', type=str,
                        help='Directory to scan for C++ files (required for CLI mode)')

    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file path (default: results_<timestamp>.json)')

    parser.add_argument('--report', '-r', type=str,
                        help='Generate text report at specified path')

    parser.add_argument('--base-model', type=str,
                        default='codellama/CodeLlama-7b-hf',
                        help='Base model for scoring (default: codellama/CodeLlama-7b-hf)')

    parser.add_argument('--mask-model', type=str,
                        default='Salesforce/codet5p-770m',
                        help='Mask filling model (default: Salesforce/codet5p-770m)')

    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'],
                        help='Device to use (default: auto-detect)')

    parser.add_argument('--n-perturbations', type=int, default=50,
                        help='Number of perturbations per sample (default: 50)')

    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for processing (default: 10)')

    parser.add_argument('--min-size', type=int, default=100,
                        help='Minimum file size in bytes (default: 100)')

    parser.add_argument('--max-size', type=int, default=100000,
                        help='Maximum file size in bytes (default: 100000)')

    parser.add_argument('--max-files', type=int, default=0,
                        help='Maximum number of files to process, 0 for all (default: 0)')

    parser.add_argument('--device-info', action='store_true',
                        help='Show device information and exit')

    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with full tracebacks')

    args = parser.parse_args()

    # Show device info if requested
    if args.device_info:
        show_device_info()
        return

    # Run in appropriate mode
    if args.cli:
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
