"""
Batch Processor and JSON Output Module
Processes multiple C++ files and exports results to JSON.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from tqdm import tqdm

from file_scanner import scan_cpp_files
from cpp_detector import CppCodeDetector, GPUManager


class BatchProcessor:
    """Processes multiple C++ files and generates reports."""

    def __init__(self,
                 base_model_name: str = "codellama/CodeLlama-7b-hf",
                 mask_filling_model_name: str = "Salesforce/codet5p-770m",
                 device: Optional[str] = None,
                 batch_size: int = 10,
                 n_perturbations: int = 50):
        """
        Initialize the batch processor.

        Args:
            base_model_name: Base model for scoring
            mask_filling_model_name: Mask filling model
            device: Device to use (None for auto-detect)
            batch_size: Batch size for processing
            n_perturbations: Number of perturbations per sample
        """
        self.detector = CppCodeDetector(
            base_model_name=base_model_name,
            mask_filling_model_name=mask_filling_model_name,
            device=device,
            batch_size=batch_size,
            n_perturbations=n_perturbations
        )

    def process_directory(self,
                          root_path: str,
                          output_path: Optional[str] = None,
                          min_size: int = 100,
                          max_size: int = 100000,
                          max_files: Optional[int] = None) -> Dict:
        """
        Process all C++ files in a directory.

        Args:
            root_path: Root directory to scan
            output_path: Path to save JSON output (optional)
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            max_files: Maximum number of files to process (None for all)

        Returns:
            Dictionary with all results
        """
        logger.info(f"Scanning directory: {root_path}")

        # Scan for C++ files
        scan_result = scan_cpp_files(
            root_path,
            organize=True,
            min_size=min_size,
            max_size=max_size
        )

        if scan_result['total_files'] == 0:
            logger.warning("No C++ files found in directory")
            return {
                'error': 'No C++ files found',
                'root_path': root_path,
                'timestamp': datetime.now().isoformat()
            }

        logger.info(f"Found {scan_result['total_files']} C++ files in {scan_result['project_count']} projects")

        # Load models once
        logger.info("Loading detection models...")
        self.detector.load_models()

        # Process files
        start_time = time.time()
        results_by_project = {}

        files_processed = 0
        for project_name, files in scan_result['projects'].items():
            logger.info(f"\nProcessing project: {project_name} ({len(files)} files)")

            project_results = []

            for file_info in tqdm(files, desc=f"Processing {project_name}"):
                if max_files and files_processed >= max_files:
                    logger.info(f"Reached maximum file limit: {max_files}")
                    break

                try:
                    # Detect AI-generated code
                    detection_result = self.detector.detect_single(file_info['content'])

                    # Combine file info and detection result
                    combined_result = {
                        'file_path': file_info['file_path'],
                        'relative_path': file_info['relative_path'],
                        'file_name': file_info['file_name'],
                        'project': project_name,
                        'detection': detection_result,
                        'timestamp': datetime.now().isoformat()
                    }

                    project_results.append(combined_result)
                    files_processed += 1

                except Exception as e:
                    logger.error(f"Error processing {file_info['file_path']}: {e}")
                    project_results.append({
                        'file_path': file_info['file_path'],
                        'relative_path': file_info['relative_path'],
                        'file_name': file_info['file_name'],
                        'project': project_name,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })

            results_by_project[project_name] = project_results

            if max_files and files_processed >= max_files:
                break

        processing_time = time.time() - start_time

        # Compile final results
        final_results = self._compile_results(
            root_path=root_path,
            scan_result=scan_result,
            results_by_project=results_by_project,
            processing_time=processing_time,
            files_processed=files_processed
        )

        # Save to JSON if output path specified
        if output_path:
            self.save_json(final_results, output_path)

        # Unload models to free memory
        self.detector.unload_models()

        return final_results

    def _compile_results(self,
                        root_path: str,
                        scan_result: Dict,
                        results_by_project: Dict,
                        processing_time: float,
                        files_processed: int) -> Dict:
        """
        Compile all results into a structured format.

        Args:
            root_path: Root directory path
            scan_result: Results from file scanning
            results_by_project: Detection results organized by project
            processing_time: Total processing time
            files_processed: Number of files processed

        Returns:
            Compiled results dictionary
        """
        # Calculate statistics
        all_detections = []
        for project_results in results_by_project.values():
            for result in project_results:
                if 'detection' in result:
                    all_detections.append(result['detection'])

        ai_generated_count = sum(1 for d in all_detections
                                 if d.get('prediction') == 'likely_ai_generated')
        possibly_ai_count = sum(1 for d in all_detections
                               if d.get('prediction') == 'possibly_ai_generated')
        human_written_count = sum(1 for d in all_detections
                                  if d.get('prediction') == 'likely_human_written')

        # Get device info
        device_info = GPUManager.get_device_info()

        return {
            'metadata': {
                'root_path': root_path,
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': round(processing_time, 2),
                'files_scanned': scan_result['total_files'],
                'files_processed': files_processed,
                'projects_count': len(results_by_project),
                'device_info': device_info,
                'detector_config': {
                    'base_model': self.detector.base_model_name,
                    'mask_filling_model': self.detector.mask_filling_model_name,
                    'device': self.detector.device,
                    'n_perturbations': self.detector.n_perturbations,
                    'batch_size': self.detector.batch_size
                }
            },
            'summary': {
                'total_analyzed': len(all_detections),
                'likely_ai_generated': ai_generated_count,
                'possibly_ai_generated': possibly_ai_count,
                'likely_human_written': human_written_count,
                'errors': files_processed - len(all_detections),
                'ai_percentage': round(100 * ai_generated_count / len(all_detections), 2)
                                if all_detections else 0,
            },
            'projects': results_by_project,
            'scan_info': {
                'total_files_found': scan_result['total_files'],
                'project_count': scan_result['project_count']
            }
        }

    @staticmethod
    def save_json(data: Dict, output_path: str, indent: int = 2):
        """
        Save results to JSON file.

        Args:
            data: Data to save
            output_path: Output file path
            indent: JSON indentation level
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to: {output_path}")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        logger.info(f"Results saved successfully ({output_file.stat().st_size} bytes)")

    @staticmethod
    def load_json(input_path: str) -> Dict:
        """
        Load results from JSON file.

        Args:
            input_path: Input file path

        Returns:
            Loaded data dictionary
        """
        logger.info(f"Loading results from: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def generate_report(self, results: Dict, output_path: Optional[str] = None) -> str:
        """
        Generate a human-readable text report.

        Args:
            results: Results dictionary
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("C++ AI-Generated Code Detection Report")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Metadata
        metadata = results['metadata']
        report_lines.append(f"Root Path: {metadata['root_path']}")
        report_lines.append(f"Timestamp: {metadata['timestamp']}")
        report_lines.append(f"Processing Time: {metadata['processing_time_seconds']} seconds")
        report_lines.append(f"Device: {metadata['detector_config']['device']}")
        report_lines.append("")

        # Summary
        summary = results['summary']
        report_lines.append("Summary:")
        report_lines.append(f"  Total Files Analyzed: {summary['total_analyzed']}")
        report_lines.append(f"  Likely AI-Generated: {summary['likely_ai_generated']} ({summary['ai_percentage']}%)")
        report_lines.append(f"  Possibly AI-Generated: {summary['possibly_ai_generated']}")
        report_lines.append(f"  Likely Human-Written: {summary['likely_human_written']}")
        report_lines.append(f"  Errors: {summary['errors']}")
        report_lines.append("")

        # Projects
        report_lines.append("Projects:")
        for project_name, project_results in results['projects'].items():
            report_lines.append(f"\n  {project_name}:")

            ai_count = sum(1 for r in project_results
                          if r.get('detection', {}).get('prediction') == 'likely_ai_generated')
            report_lines.append(f"    Files: {len(project_results)}")
            report_lines.append(f"    Likely AI-Generated: {ai_count}")

            # List suspicious files
            if ai_count > 0:
                report_lines.append("    Suspicious files:")
                for result in project_results:
                    if result.get('detection', {}).get('prediction') == 'likely_ai_generated':
                        score = result['detection'].get('detectcodegpt_score', 0)
                        report_lines.append(f"      - {result['relative_path']} (score: {score:.4f})")

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_path}")

        return report_text
