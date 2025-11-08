"""
C++ Code Detector Module
Adapts DetectCodeGPT to detect AI-generated C++ code.
"""

import sys
import os
import torch
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from tqdm import tqdm
import functools

# Add parent directory to path to import from code-detection
sys.path.insert(0, str(Path(__file__).parent.parent / "code-detection"))

from baselines.utils.loadmodel import load_base_model_and_tokenizer, load_mask_filling_model
from baselines.utils.run_baseline import get_roc_metrics
from baselines.loss import get_ll, get_lls
from baselines.rank import get_rank, get_ranks


class GPUManager:
    """Manages GPU availability and device selection."""

    @staticmethod
    def get_device() -> str:
        """
        Detect and return the best available device.

        Returns:
            Device string: 'cuda', 'mps' (for Mac), or 'cpu'
        """
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA is available with {gpu_count} GPU(s)")
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        elif torch.backends.mps.is_available():
            logger.info("MPS (Metal Performance Shaders) is available")
            return 'mps'
        else:
            logger.warning("No GPU available, using CPU")
            return 'cpu'

    @staticmethod
    def get_device_info() -> Dict:
        """
        Get detailed information about available devices.

        Returns:
            Dictionary with device information
        """
        info = {
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'device_count': 0,
            'devices': []
        }

        if torch.cuda.is_available():
            info['device_count'] = torch.cuda.device_count()
            for i in range(info['device_count']):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory': torch.cuda.get_device_properties(i).total_memory,
                    'capability': torch.cuda.get_device_capability(i)
                }
                info['devices'].append(device_info)

        return info


class CppCodeDetector:
    """Detector for AI-generated C++ code using DetectCodeGPT."""

    def __init__(self,
                 base_model_name: str = "codellama/CodeLlama-7b-hf",
                 mask_filling_model_name: str = "Salesforce/codet5p-770m",
                 device: Optional[str] = None,
                 batch_size: int = 10,
                 n_perturbations: int = 50,
                 pct_words_masked: float = 0.5):
        """
        Initialize the C++ code detector.

        Args:
            base_model_name: Name of the base model for scoring
            mask_filling_model_name: Name of the mask filling model
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            batch_size: Batch size for processing
            n_perturbations: Number of perturbations to generate
            pct_words_masked: Percentage of words to mask during perturbation
        """
        self.base_model_name = base_model_name
        self.mask_filling_model_name = mask_filling_model_name
        self.device = device if device else GPUManager.get_device()
        self.batch_size = batch_size
        self.n_perturbations = n_perturbations
        self.pct_words_masked = pct_words_masked

        # Model config will be loaded lazily
        self.model_config = None
        self.models_loaded = False

        logger.info(f"Initialized CppCodeDetector with device: {self.device}")

    def load_models(self):
        """Load the models for detection."""
        if self.models_loaded:
            return

        logger.info("Loading models...")

        # Create args object similar to main.py
        class Args:
            pass

        args = Args()
        args.mask_filling_model_name = self.mask_filling_model_name
        args.base_model_name = self.base_model_name
        args.DEVICE = self.device
        args.cache_dir = "~/.cache/huggingface/hub"
        args.int8 = False
        args.half = False
        args.base_half = False
        args.mask_top_p = 1.0
        args.mask_temperature = 1.0
        args.batch_size = self.batch_size
        args.chunk_size = 10
        args.span_length = 2
        args.buffer_size = 1
        args.pct_words_masked = self.pct_words_masked
        args.perturb_type = "random-insert-space+newline"
        args.n_perturbation_rounds = 1

        self.args = args
        self.model_config = {}
        self.model_config['cache_dir'] = args.cache_dir

        # Load mask filling model
        logger.info(f"Loading mask filling model: {self.mask_filling_model_name}")
        self.model_config = load_mask_filling_model(args, self.mask_filling_model_name, self.model_config)

        # Load base scoring model
        logger.info(f"Loading base scoring model: {self.base_model_name}")
        # Move mask model to CPU to free GPU memory
        self.model_config['mask_model'] = self.model_config['mask_model'].cpu()
        torch.cuda.empty_cache()

        self.model_config = load_base_model_and_tokenizer(args, self.model_config)

        self.models_loaded = True
        logger.info("Models loaded successfully")

    def _preprocess_cpp_code(self, code: str, max_tokens: int = 128) -> str:
        """
        Preprocess C++ code for detection.

        Args:
            code: C++ source code
            max_tokens: Maximum number of tokens to keep

        Returns:
            Preprocessed code
        """
        # Remove excessive whitespace but preserve structure
        lines = code.split('\n')
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        code = '\n'.join(lines)

        # Truncate to max_tokens words (rough approximation)
        words = code.split()
        if len(words) > max_tokens:
            code = ' '.join(words[:max_tokens])

        return code

    def detect_single(self, code: str) -> Dict:
        """
        Detect if a single code snippet is AI-generated.

        Args:
            code: C++ code to analyze

        Returns:
            Dictionary with detection results including scores and probabilities
        """
        if not self.models_loaded:
            self.load_models()

        # Preprocess code
        code = self._preprocess_cpp_code(code)

        # Import perturbation functions
        from main import perturb_texts

        # Generate perturbations
        perturb_fn = functools.partial(perturb_texts, args=self.args, model_config=self.model_config)

        logger.info(f"Generating {self.n_perturbations} perturbations...")
        perturbed_texts = perturb_fn([code for _ in range(self.n_perturbations)])

        # Calculate log likelihood
        logger.info("Computing log likelihood...")
        original_ll = get_ll(code, self.args, self.model_config)

        # Calculate log rank
        logger.info("Computing log rank...")
        original_logrank = get_rank(code, self.args, self.model_config, log=True)

        # Calculate perturbed log likelihoods
        logger.info("Computing perturbed log likelihoods...")
        perturbed_lls = get_lls(perturbed_texts, self.args, self.model_config)
        perturbed_lls_clean = [ll for ll in perturbed_lls if not math.isnan(ll)]

        perturbed_ll_mean = np.mean(perturbed_lls_clean) if perturbed_lls_clean else 0
        perturbed_ll_std = np.std(perturbed_lls_clean) if len(perturbed_lls_clean) > 1 else 1

        # Calculate perturbed log ranks
        logger.info("Computing perturbed log ranks...")
        perturbed_logranks = get_ranks(perturbed_texts, self.args, self.model_config, log=True)
        perturbed_logranks_clean = [lr for lr in perturbed_logranks if not math.isnan(lr)]

        perturbed_logrank_mean = np.mean(perturbed_logranks_clean) if perturbed_logranks_clean else 0

        # Calculate detection scores
        # DetectGPT score
        detectgpt_score = (original_ll - perturbed_ll_mean) / perturbed_ll_std if perturbed_ll_std > 0 else 0

        # Log Rank Ratio (LRR)
        lrr_score = -original_ll / original_logrank if original_logrank != 0 else 0

        # DetectCodeGPT score (NPR - Normalized Perturbation Rank)
        detectcodegpt_score = perturbed_logrank_mean / original_logrank if original_logrank != 0 else 0

        result = {
            'code_length': len(code),
            'word_count': len(code.split()),
            'line_count': len(code.splitlines()),
            'original_ll': float(original_ll),
            'original_logrank': float(original_logrank),
            'perturbed_ll_mean': float(perturbed_ll_mean),
            'perturbed_ll_std': float(perturbed_ll_std),
            'perturbed_logrank_mean': float(perturbed_logrank_mean),
            'detectgpt_score': float(detectgpt_score),
            'lrr_score': float(lrr_score),
            'detectcodegpt_score': float(detectcodegpt_score),
            'n_perturbations': self.n_perturbations,
            'n_valid_perturbations': len(perturbed_lls_clean),
        }

        # Add interpretation (higher scores suggest AI-generated)
        # These thresholds are rough estimates and should be calibrated
        if detectcodegpt_score > 1.0:
            result['prediction'] = 'likely_ai_generated'
            result['confidence'] = 'high'
        elif detectcodegpt_score > 0.95:
            result['prediction'] = 'possibly_ai_generated'
            result['confidence'] = 'medium'
        else:
            result['prediction'] = 'likely_human_written'
            result['confidence'] = 'medium'

        return result

    def detect_batch(self, codes: List[str], show_progress: bool = True) -> List[Dict]:
        """
        Detect AI-generated code for a batch of code snippets.

        Args:
            codes: List of C++ code strings
            show_progress: Whether to show progress bar

        Returns:
            List of detection result dictionaries
        """
        results = []

        iterator = tqdm(codes, desc="Detecting AI-generated code") if show_progress else codes

        for code in iterator:
            try:
                result = self.detect_single(code)
                results.append(result)
            except Exception as e:
                logger.error(f"Error detecting code: {e}")
                results.append({
                    'error': str(e),
                    'prediction': 'error',
                    'confidence': 'none'
                })

        return results

    def unload_models(self):
        """Unload models to free memory."""
        if self.model_config:
            if 'mask_model' in self.model_config:
                del self.model_config['mask_model']
            if 'base_model' in self.model_config:
                del self.model_config['base_model']

        torch.cuda.empty_cache()
        self.models_loaded = False
        logger.info("Models unloaded")
