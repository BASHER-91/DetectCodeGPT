"""
GUI Application for C++ AI-Generated Code Detection
Provides a user-friendly interface for scanning and analyzing C++ projects.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys

from batch_processor import BatchProcessor
from cpp_detector import GPUManager


class DetectorGUI:
    """Main GUI application for C++ code detection."""

    def __init__(self, root):
        """
        Initialize the GUI.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("DetectCodeGPT - C++ AI Code Detector")
        self.root.geometry("1000x800")

        # Variables
        self.directory_var = tk.StringVar()
        self.output_path_var = tk.StringVar(value="results.json")
        self.base_model_var = tk.StringVar(value="codellama/CodeLlama-7b-hf")
        self.mask_model_var = tk.StringVar(value="Salesforce/codet5p-770m")
        self.device_var = tk.StringVar(value="auto")
        self.n_perturbations_var = tk.IntVar(value=50)
        self.batch_size_var = tk.IntVar(value=10)
        self.min_size_var = tk.IntVar(value=100)
        self.max_size_var = tk.IntVar(value=100000)
        self.max_files_var = tk.IntVar(value=0)  # 0 means no limit

        self.processing = False
        self.processor = None
        self.results = None

        # Setup GUI
        self._create_widgets()
        self._setup_logging()

        # Display device info
        self._display_device_info()

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 1: Configuration
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text='Configuration')
        self._create_config_tab()

        # Tab 2: Processing
        self.process_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.process_frame, text='Processing')
        self._create_process_tab()

        # Tab 3: Results
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text='Results')
        self._create_results_tab()

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_config_tab(self):
        """Create configuration tab widgets."""
        # Directory selection
        dir_frame = ttk.LabelFrame(self.config_frame, text="Directory Selection", padding=10)
        dir_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(dir_frame, text="C++ Project Directory:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(dir_frame, textvariable=self.directory_var, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse...", command=self._browse_directory).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(dir_frame, text="Output JSON Path:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(dir_frame, textvariable=self.output_path_var, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(dir_frame, text="Browse...", command=self._browse_output).grid(row=1, column=2, padx=5, pady=5)

        # Model configuration
        model_frame = ttk.LabelFrame(self.config_frame, text="Model Configuration", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(model_frame, text="Base Model:").grid(row=0, column=0, sticky='w', pady=5)
        base_model_combo = ttk.Combobox(model_frame, textvariable=self.base_model_var, width=57)
        base_model_combo['values'] = [
            'codellama/CodeLlama-7b-hf',
            'codellama/CodeLlama-13b-hf',
            'Salesforce/codegen-2B-mono',
            'microsoft/CodeGPT-small-py'
        ]
        base_model_combo.grid(row=0, column=1, padx=5, pady=5, columnspan=2)

        ttk.Label(model_frame, text="Mask Filling Model:").grid(row=1, column=0, sticky='w', pady=5)
        mask_model_combo = ttk.Combobox(model_frame, textvariable=self.mask_model_var, width=57)
        mask_model_combo['values'] = [
            'Salesforce/codet5p-770m',
            'Salesforce/codet5-base',
            'Salesforce/CodeT5-large'
        ]
        mask_model_combo.grid(row=1, column=1, padx=5, pady=5, columnspan=2)

        ttk.Label(model_frame, text="Device:").grid(row=2, column=0, sticky='w', pady=5)
        device_combo = ttk.Combobox(model_frame, textvariable=self.device_var, width=20)
        device_combo['values'] = ['auto', 'cuda', 'cpu', 'mps']
        device_combo.grid(row=2, column=1, sticky='w', padx=5, pady=5)

        # Detection parameters
        param_frame = ttk.LabelFrame(self.config_frame, text="Detection Parameters", padding=10)
        param_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(param_frame, text="Number of Perturbations:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Spinbox(param_frame, from_=10, to=200, textvariable=self.n_perturbations_var, width=20).grid(
            row=0, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(param_frame, text="Batch Size:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Spinbox(param_frame, from_=1, to=50, textvariable=self.batch_size_var, width=20).grid(
            row=1, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(param_frame, text="Min File Size (bytes):").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Spinbox(param_frame, from_=0, to=10000, textvariable=self.min_size_var, width=20).grid(
            row=2, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(param_frame, text="Max File Size (bytes):").grid(row=3, column=0, sticky='w', pady=5)
        ttk.Spinbox(param_frame, from_=1000, to=10000000, textvariable=self.max_size_var, width=20).grid(
            row=3, column=1, sticky='w', padx=5, pady=5)

        ttk.Label(param_frame, text="Max Files to Process (0=all):").grid(row=4, column=0, sticky='w', pady=5)
        ttk.Spinbox(param_frame, from_=0, to=10000, textvariable=self.max_files_var, width=20).grid(
            row=4, column=1, sticky='w', padx=5, pady=5)

        # Device info
        self.device_info_frame = ttk.LabelFrame(self.config_frame, text="Device Information", padding=10)
        self.device_info_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.device_info_text = scrolledtext.ScrolledText(self.device_info_frame, height=6, wrap=tk.WORD)
        self.device_info_text.pack(fill='both', expand=True)

    def _create_process_tab(self):
        """Create processing tab widgets."""
        # Control buttons
        button_frame = ttk.Frame(self.process_frame)
        button_frame.pack(fill='x', padx=10, pady=10)

        self.start_button = ttk.Button(button_frame, text="Start Detection", command=self._start_detection,
                                       style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self._stop_detection, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=5)

        # Progress bar
        progress_frame = ttk.Frame(self.process_frame)
        progress_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(side=tk.LEFT, fill='x', expand=True, padx=5)

        # Log output
        log_frame = ttk.LabelFrame(self.process_frame, text="Processing Log", padding=5)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state='disabled')
        self.log_text.pack(fill='both', expand=True)

    def _create_results_tab(self):
        """Create results tab widgets."""
        # Results control buttons
        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(button_frame, text="Load Results", command=self._load_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Report", command=self._export_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self._clear_results).pack(side=tk.LEFT, padx=5)

        # Results display
        results_display_frame = ttk.LabelFrame(self.results_frame, text="Detection Results", padding=5)
        results_display_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_display_frame, wrap=tk.WORD, state='disabled')
        self.results_text.pack(fill='both', expand=True)

    def _browse_directory(self):
        """Browse for directory to scan."""
        directory = filedialog.askdirectory(title="Select C++ Project Directory")
        if directory:
            self.directory_var.set(directory)

    def _browse_output(self):
        """Browse for output file path."""
        filepath = filedialog.asksaveasfilename(
            title="Save Results As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            self.output_path_var.set(filepath)

    def _display_device_info(self):
        """Display device information."""
        device_info = GPUManager.get_device_info()
        info_text = f"CUDA Available: {device_info['cuda_available']}\n"
        info_text += f"MPS Available: {device_info['mps_available']}\n"
        info_text += f"Device Count: {device_info['device_count']}\n\n"

        if device_info['devices']:
            info_text += "Available GPUs:\n"
            for device in device_info['devices']:
                info_text += f"  GPU {device['id']}: {device['name']}\n"
                info_text += f"    Memory: {device['total_memory'] / (1024**3):.2f} GB\n"
                info_text += f"    Capability: {device['capability']}\n"
        else:
            info_text += "No GPUs available. Will use CPU.\n"

        self.device_info_text.delete(1.0, tk.END)
        self.device_info_text.insert(1.0, info_text)

    def _setup_logging(self):
        """Setup logging to GUI."""
        # Create a custom handler to redirect logs to GUI
        class GUILogHandler:
            def __init__(self, text_widget, root):
                self.text_widget = text_widget
                self.root = root

            def write(self, message):
                if message.strip():  # Avoid empty lines
                    self.text_widget.configure(state='normal')
                    self.text_widget.insert(tk.END, message + '\n')
                    self.text_widget.see(tk.END)
                    self.text_widget.configure(state='disabled')
                    self.root.update_idletasks()

        self.gui_log_handler = GUILogHandler(self.log_text, self.root)

    def _log_message(self, message):
        """Log a message to the GUI."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        self.root.update_idletasks()

    def _clear_log(self):
        """Clear the log display."""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')

    def _start_detection(self):
        """Start the detection process."""
        # Validate inputs
        directory = self.directory_var.get()
        if not directory or not Path(directory).exists():
            messagebox.showerror("Error", "Please select a valid directory")
            return

        output_path = self.output_path_var.get()
        if not output_path:
            messagebox.showerror("Error", "Please specify an output path")
            return

        # Update UI state
        self.processing = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_bar.start()
        self.status_bar.config(text="Processing...")

        # Start processing in a separate thread
        thread = threading.Thread(target=self._run_detection, daemon=True)
        thread.start()

    def _run_detection(self):
        """Run detection process (called in separate thread)."""
        try:
            self._log_message("Starting detection process...")

            # Get device
            device = None if self.device_var.get() == 'auto' else self.device_var.get()

            # Create processor
            self.processor = BatchProcessor(
                base_model_name=self.base_model_var.get(),
                mask_filling_model_name=self.mask_model_var.get(),
                device=device,
                batch_size=self.batch_size_var.get(),
                n_perturbations=self.n_perturbations_var.get()
            )

            # Process directory
            max_files = self.max_files_var.get() if self.max_files_var.get() > 0 else None

            self.results = self.processor.process_directory(
                root_path=self.directory_var.get(),
                output_path=self.output_path_var.get(),
                min_size=self.min_size_var.get(),
                max_size=self.max_size_var.get(),
                max_files=max_files
            )

            self._log_message("Detection complete!")
            self._log_message(f"Results saved to: {self.output_path_var.get()}")

            # Display results
            self.root.after(0, self._display_results)

            # Show completion message
            self.root.after(0, lambda: messagebox.showinfo("Complete", "Detection completed successfully!"))

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            self._log_message(f"ERROR: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {str(e)}"))

        finally:
            # Update UI state
            self.root.after(0, self._finish_detection)

    def _stop_detection(self):
        """Stop the detection process."""
        # Note: This is a simple implementation. For proper cancellation,
        # you'd need to implement cooperative cancellation in the processor
        self.processing = False
        self._log_message("Stopping detection...")
        self._finish_detection()

    def _finish_detection(self):
        """Clean up after detection finishes."""
        self.processing = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.progress_bar.stop()
        self.status_bar.config(text="Ready")

    def _display_results(self):
        """Display results in the results tab."""
        if not self.results:
            return

        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)

        # Format and display results
        summary = self.results.get('summary', {})
        metadata = self.results.get('metadata', {})

        output = "=" * 80 + "\n"
        output += "DETECTION RESULTS SUMMARY\n"
        output += "=" * 80 + "\n\n"

        output += f"Root Path: {metadata.get('root_path', 'N/A')}\n"
        output += f"Timestamp: {metadata.get('timestamp', 'N/A')}\n"
        output += f"Processing Time: {metadata.get('processing_time_seconds', 0)} seconds\n"
        output += f"Device: {metadata.get('detector_config', {}).get('device', 'N/A')}\n\n"

        output += "Summary:\n"
        output += f"  Total Files Analyzed: {summary.get('total_analyzed', 0)}\n"
        output += f"  Likely AI-Generated: {summary.get('likely_ai_generated', 0)}\n"
        output += f"  Possibly AI-Generated: {summary.get('possibly_ai_generated', 0)}\n"
        output += f"  Likely Human-Written: {summary.get('likely_human_written', 0)}\n"
        output += f"  AI Percentage: {summary.get('ai_percentage', 0)}%\n\n"

        # Projects
        output += "Projects:\n"
        for project_name, project_results in self.results.get('projects', {}).items():
            ai_count = sum(1 for r in project_results
                          if r.get('detection', {}).get('prediction') == 'likely_ai_generated')
            output += f"\n  {project_name}: {len(project_results)} files, {ai_count} likely AI-generated\n"

            # List suspicious files
            for result in project_results:
                if result.get('detection', {}).get('prediction') == 'likely_ai_generated':
                    score = result.get('detection', {}).get('detectcodegpt_score', 0)
                    output += f"    - {result['relative_path']} (score: {score:.4f})\n"

        self.results_text.insert(1.0, output)
        self.results_text.configure(state='disabled')

        # Switch to results tab
        self.notebook.select(self.results_frame)

    def _load_results(self):
        """Load results from a JSON file."""
        filepath = filedialog.askopenfilename(
            title="Load Results",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.results = json.load(f)
                self._display_results()
                messagebox.showinfo("Success", "Results loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load results: {str(e)}")

    def _export_report(self):
        """Export a text report."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filepath:
            try:
                processor = BatchProcessor()
                report = processor.generate_report(self.results, output_path=filepath)
                messagebox.showinfo("Success", f"Report exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export report: {str(e)}")

    def _clear_results(self):
        """Clear the results display."""
        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.configure(state='disabled')
        self.results = None


def main():
    """Main entry point for GUI application."""
    root = tk.Tk()

    # Set style
    style = ttk.Style()
    style.theme_use('clam')

    # Create and run GUI
    app = DetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
