"""
GPU utilities for M1 Mac GPU acceleration
Provides GPU detection, configuration, and optimization functions
"""

import os
import platform
from typing import Dict, Any, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class GPUManager:
    """Manages GPU resources and configuration for M1 Mac"""
    
    def __init__(self):
        self._gpu_available = False
        self._device = "cpu"
        self._platform = platform.system()
        self._memory_limit_mb = 0
        
        # Detect GPU availability
        self._detect_gpu()
    
    def _detect_gpu(self):
        """Detect available GPU"""
        if TORCH_AVAILABLE:
            # Check for M1 GPU (Metal Performance Shaders)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._gpu_available = True
                self._device = "mps"
                # M1 GPU memory limit (approximate)
                self._memory_limit_mb = 8192  # 8GB for M1
            # Check for CUDA
            elif torch.cuda.is_available():
                self._gpu_available = True
                self._device = "cuda"
                self._memory_limit_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        else:
            # If PyTorch not available, assume CPU
            self._gpu_available = False
            self._device = "cpu"
            
            # Try to get system memory for CPU
            if PSUTIL_AVAILABLE:
                self._memory_limit_mb = psutil.virtual_memory().total // (1024 * 1024)
    
    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available"""
        return self._gpu_available
    
    @property
    def device(self) -> str:
        """Get the device string (mps, cuda, or cpu)"""
        return self._device
    
    @property
    def platform(self) -> str:
        """Get the platform name"""
        return self._platform
    
    @property
    def memory_limit_mb(self) -> int:
        """Get memory limit in MB"""
        return self._memory_limit_mb
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return {
            "device": self._device,
            "gpu_available": self._gpu_available,
            "platform": self._platform,
            "memory_limit_mb": self._memory_limit_mb
        }
    
    def display_memory_info(self):
        """Display memory information"""
        from utils.logging import get_logger
        logger = get_logger(__name__)
        
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            logger.info(f"Memory - Total: {memory.total // (1024**3)}GB, "
                       f"Available: {memory.available // (1024**3)}GB, "
                       f"Used: {memory.percent}%")
        
        if self._gpu_available and TORCH_AVAILABLE:
            if self._device == "mps":
                logger.info("M1 GPU (Metal) is available")
            elif self._device == "cuda":
                logger.info(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB")
    
    def create_paddleocr_config(self) -> Dict[str, Any]:
        """Create PaddleOCR configuration optimized for M1 GPU"""
        config = {
            "use_angle_cls": True,
            "lang": "en",
            "use_gpu": self._gpu_available and self._device != "cpu",
            "show_log": False
        }
        
        # M1 Mac specific optimizations
        if self._device == "mps":
            # PaddleOCR doesn't directly support MPS, but we can use CPU with optimizations
            config["use_gpu"] = False
            config["cpu_threads"] = os.cpu_count()
            config["enable_mkldnn"] = False
        
        return config
    
    def optimize_for_m1(self):
        """Apply M1-specific optimizations"""
        # Set environment variables for optimal performance
        if self._platform == "Darwin":  # macOS
            # Optimize thread usage for M1
            cpu_count = os.cpu_count() or 4
            os.environ['OMP_NUM_THREADS'] = str(cpu_count)
            os.environ['MKL_NUM_THREADS'] = str(cpu_count)
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_count)
            
            # Disable OpenMP warnings
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get or create the global GPU manager instance"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager


def is_gpu_available() -> bool:
    """Check if GPU is available"""
    return get_gpu_manager().gpu_available


def get_optimal_device() -> str:
    """Get the optimal device for computation"""
    manager = get_gpu_manager()
    return manager.device


def show_memory_usage():
    """Display current memory usage"""
    if PSUTIL_AVAILABLE:
        from utils.logging import get_logger
        logger = get_logger(__name__)
        
        memory = psutil.virtual_memory()
        logger.debug(f"Memory Usage - Used: {memory.percent}%, "
                    f"Available: {memory.available // (1024**3)}GB / "
                    f"{memory.total // (1024**3)}GB")
    else:
        # Fallback: just log that memory info is not available
        from utils.logging import get_logger
        logger = get_logger(__name__)
        logger.debug("Memory usage info not available (psutil not installed)")

