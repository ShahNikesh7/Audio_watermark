"""
Perceptual Significance Metric for Adaptive Bit Allocation
Task 1.2.2: Design Adaptive Bit Allocation
Micro Task 1.2.2.1: Develop a perceptual significance metric for different frequency bands
Micro Task 1.2.2.2: Design an algorithm to dynamically allocate watermark bits based on the significance metric

This module implements:
1. Perceptual significance metric based on psychoacoustic masking thresholds
2. Intelligent dynamic bit allocation algorithm that optimally distributes watermark bits
   across frequency bands based on their perceptual significance

Task 1.2.2.2 Enhancement: Advanced Dynamic Bit Allocation Algorithm
- Multi-objective optimization considering robustness, capacity, and imperceptibility
- Adaptive constraint handling with dynamic band capabilities
- Intelligent reallocation strategies with temporal consistency
- Context-aware allocation with audio content analysis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import signal
from scipy.interpolate import interp1d

from .moore_glasberg import MooreGlasbergAnalyzer, PerceptualAnalyzer
from .band_thresholds import BandThresholdCalculator

logger = logging.getLogger(__name__)


class PerceptualSignificanceMetric:
    """
    Computes perceptual significance for frequency bands based on masking thresholds.
    
    Higher masking thresholds indicate that the band can tolerate more watermark energy
    without being perceived, thus having higher perceptual significance for watermarking.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 1000,
                 hop_length: int = 512,
                 n_critical_bands: int = 24,
                 significance_method: str = "logarithmic",
                 temporal_averaging: bool = True,
                 adaptive_normalization: bool = True):
        """
        Initialize the perceptual significance metric calculator.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size for frequency analysis
            hop_length: Hop length for STFT
            n_critical_bands: Number of critical bands
            significance_method: Method for computing significance ("linear", "logarithmic", "power")
            temporal_averaging: Whether to average significance across time frames
            adaptive_normalization: Whether to use adaptive normalization
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_critical_bands = n_critical_bands
        self.significance_method = significance_method
        self.temporal_averaging = temporal_averaging
        self.adaptive_normalization = adaptive_normalization
        
        # Initialize psychoacoustic analyzer
        self.psychoacoustic_analyzer = MooreGlasbergAnalyzer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_critical_bands=n_critical_bands
        )
        
        # Initialize band threshold calculator
        self.band_calculator = BandThresholdCalculator(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_bands=n_critical_bands
        )
        
        # Frequency band characteristics
        self.frequencies = self.psychoacoustic_analyzer.frequencies
        self.band_indices = self.psychoacoustic_analyzer.band_indices
        self.band_centers = self.psychoacoustic_analyzer.band_centers
        
        # Cache for efficiency
        self._significance_cache = {}
        
        logger.info(f"Initialized Perceptual Significance Metric with {n_critical_bands} bands, "
                   f"method: {significance_method}")
    
    def compute_band_significance(self, 
                                 audio: np.ndarray,
                                 return_temporal: bool = False) -> Dict[str, np.ndarray]:
        """
        Compute perceptual significance for each critical band.
        
        Args:
            audio: Input audio signal
            return_temporal: Whether to return temporal evolution of significance
            
        Returns:
            Dictionary containing significance metrics per band
        """
        # Get masking analysis from psychoacoustic model
        masking_analysis = self.psychoacoustic_analyzer.analyze_masking_threshold(audio)
        band_thresholds = masking_analysis['band_thresholds']  # (n_bands, n_frames)
        
        # Compute significance for each frame
        temporal_significance = self._compute_significance_temporal(band_thresholds)
        
        # Temporal averaging if requested
        if self.temporal_averaging:
            # Use weighted average emphasizing frames with higher energy
            power_spectrum = masking_analysis['power_spectrum']
            frame_weights = np.mean(power_spectrum, axis=0)  # Energy per frame
            frame_weights = frame_weights / (np.sum(frame_weights) + 1e-10)
            
            # Weighted average across time
            avg_significance = np.average(temporal_significance, axis=1, weights=frame_weights)
        else:
            avg_significance = np.mean(temporal_significance, axis=1)
        
        # Adaptive normalization
        if self.adaptive_normalization:
            normalized_significance = self._adaptive_normalize(avg_significance, band_thresholds)
        else:
            normalized_significance = self._standard_normalize(avg_significance)
        
        result = {
            'band_significance': normalized_significance,
            'raw_significance': avg_significance,
            'band_thresholds': band_thresholds,
            'band_centers': self.band_centers,
            'total_significance': np.sum(normalized_significance)
        }
        
        if return_temporal:
            result['temporal_significance'] = temporal_significance
        
        return result
    
    def _compute_significance_temporal(self, band_thresholds: np.ndarray) -> np.ndarray:
        """
        Compute significance values for each band and time frame.
        
        Args:
            band_thresholds: Masking thresholds per band (n_bands, n_frames)
            
        Returns:
            Significance values (n_bands, n_frames)
        """
        # Avoid numerical issues
        safe_thresholds = np.maximum(band_thresholds, 1e-10)
        
        if self.significance_method == "linear":
            # Linear relationship: higher threshold = higher significance
            significance = safe_thresholds
            
        elif self.significance_method == "logarithmic":
            # Logarithmic relationship for perceptual scaling
            significance = np.log10(safe_thresholds + 1.0)
            
        elif self.significance_method == "power":
            # Power-law relationship emphasizing high-threshold bands
            power_exponent = 0.6  # Psychoacoustic power-law exponent
            significance = np.power(safe_thresholds, power_exponent)
            
        elif self.significance_method == "db_scaled":
            # dB scaling with offset for positive values
            significance = 10 * np.log10(safe_thresholds) + 60  # Add 60dB offset
            significance = np.maximum(significance, 0)  # Ensure non-negative
            
        else:
            raise ValueError(f"Unknown significance method: {self.significance_method}")
        
        return significance
    
    def _adaptive_normalize(self, significance: np.ndarray, 
                          band_thresholds: np.ndarray) -> np.ndarray:
        """
        Adaptive normalization considering frequency-dependent characteristics.
        
        Args:
            significance: Raw significance values
            band_thresholds: Original masking thresholds for context
            
        Returns:
            Adaptively normalized significance values
        """
        # Consider frequency-dependent perceptual importance
        freq_weights = self._compute_frequency_weights()
        
        # Apply frequency weighting
        weighted_significance = significance * freq_weights
        
        # Adaptive scaling based on threshold distribution
        threshold_variance = np.var(np.mean(band_thresholds, axis=1))
        if threshold_variance > 1e-6:
            # High variance: emphasize differences more
            adaptive_power = 1.2
        else:
            # Low variance: more uniform distribution
            adaptive_power = 0.8
        
        # Apply adaptive power scaling
        normalized = np.power(weighted_significance / np.max(weighted_significance + 1e-10), 
                            adaptive_power)
        
        # Ensure sum equals number of bands (for bit allocation)
        return normalized * self.n_critical_bands / (np.sum(normalized) + 1e-10)
    
    def _standard_normalize(self, significance: np.ndarray) -> np.ndarray:
        """Standard min-max normalization."""
        min_val = np.min(significance)
        max_val = np.max(significance)
        if max_val - min_val < 1e-10:
            return np.ones_like(significance)
        return (significance - min_val) / (max_val - min_val)
    
    def _compute_frequency_weights(self) -> np.ndarray:
        """
        Compute frequency-dependent weights based on auditory perception.
        
        Returns:
            Frequency weights for each critical band
        """
        # A-weighting inspired frequency response
        band_freqs = self.band_centers
        
        # Emphasize frequencies around 1-4 kHz (speech important region)
        freq_weights = np.ones(len(band_freqs))
        
        for i, freq in enumerate(band_freqs):
            if freq > 0:
                # Simple A-weighting approximation
                if 200 <= freq <= 6000:
                    # Peak around 1-4 kHz
                    weight = 1.0 + 0.5 * np.exp(-((freq - 2000) / 1500) ** 2)
                else:
                    # Lower weight for very low and very high frequencies
                    weight = 0.7
                freq_weights[i] = weight
        
        return freq_weights / np.mean(freq_weights)  # Normalize to mean=1


class AdaptiveBitAllocator:
    """
    Adaptive bit allocation system based on perceptual significance metrics.
    
    Allocates more watermark bits to frequency bands with higher perceptual significance,
    enabling optimal trade-off between robustness and perceptual quality.
    """
    
    def __init__(self,
                 total_bits: int = 1000,
                 min_bits_per_band: int = 1,
                 max_bits_per_band: int = 8,
                 allocation_strategy: str = "proportional",
                 reserve_bits: int = 4):
        """
        Initialize adaptive bit allocator.
        
        Args:
            total_bits: Total number of watermark bits to allocate
            min_bits_per_band: Minimum bits per band
            max_bits_per_band: Maximum bits per band
            allocation_strategy: Strategy for bit allocation ("proportional", "threshold", "optimal", "dynamic")
            reserve_bits: Reserved bits for error correction overhead
        """
        self.total_bits = total_bits
        self.min_bits_per_band = min_bits_per_band
        self.max_bits_per_band = max_bits_per_band
        self.allocation_strategy = allocation_strategy
        self.reserve_bits = reserve_bits
        
        # Available bits for allocation
        self.available_bits = max(0, total_bits - reserve_bits)
        
        logger.info(f"Initialized Adaptive Bit Allocator: {total_bits} total bits, "
                   f"{self.available_bits} available for allocation")
    
    def allocate_bits(self, 
                     significance_values: np.ndarray,
                     band_capabilities: Optional[np.ndarray] = None,
                     audio_features: Optional[Dict[str, np.ndarray]] = None,
                     temporal_context: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Allocate bits across frequency bands based on perceptual significance.
        
        Args:
            significance_values: Perceptual significance for each band
            band_capabilities: Optional capacity constraints per band
            audio_features: Additional audio characteristics for dynamic allocation
            temporal_context: Previous allocation states for temporal consistency
            
        Returns:
            Dictionary with bit allocation results
        """
        n_bands = len(significance_values)
        
        if self.allocation_strategy == "proportional":
            allocation = self._proportional_allocation(significance_values)
        elif self.allocation_strategy == "threshold":
            allocation = self._threshold_allocation(significance_values)
        elif self.allocation_strategy == "optimal":
            allocation = self._optimal_allocation(significance_values, band_capabilities)
        elif self.allocation_strategy == "dynamic":
            # Task 1.2.2.2: Advanced Dynamic Bit Allocation
            return self.dynamic_allocate_bits(significance_values, audio_features, 
                                            temporal_context, band_capabilities)
        else:
            raise ValueError(f"Unknown allocation strategy: {self.allocation_strategy}")
        
        # Apply constraints
        allocation = self._apply_constraints(allocation, n_bands)
        
        # Compute allocation metrics
        allocation_efficiency = self._compute_allocation_efficiency(allocation, significance_values)
        
        return {
            'bit_allocation': allocation,
            'total_allocated': np.sum(allocation),
            'remaining_bits': self.available_bits - np.sum(allocation),
            'allocation_efficiency': allocation_efficiency,
            'utilization_ratio': np.sum(allocation) / self.available_bits,
            'band_utilization': allocation / self.max_bits_per_band
        }
    
    def _proportional_allocation(self, significance_values: np.ndarray) -> np.ndarray:
        """Proportional allocation based on significance values."""
        # Normalize significance to sum to 1
        norm_significance = significance_values / (np.sum(significance_values) + 1e-10)
        
        # Allocate bits proportionally
        raw_allocation = norm_significance * self.available_bits
        
        # Round to integers
        allocation = np.round(raw_allocation).astype(int)
        
        # Handle rounding errors
        diff = self.available_bits - np.sum(allocation)
        if diff > 0:
            # Add remaining bits to highest significance bands
            highest_indices = np.argsort(significance_values)[-diff:]
            allocation[highest_indices] += 1
        elif diff < 0:
            # Remove excess bits from lowest significance bands
            lowest_indices = np.argsort(significance_values)[:-diff]
            allocation[lowest_indices] = np.maximum(allocation[lowest_indices] - 1, 0)
        
        return allocation
    
    def _threshold_allocation(self, significance_values: np.ndarray) -> np.ndarray:
        """Threshold-based allocation with priority to high-significance bands."""
        allocation = np.zeros(len(significance_values), dtype=int)
        remaining_bits = self.available_bits
        
        # Sort bands by significance (descending)
        sorted_indices = np.argsort(significance_values)[::-1]
        
        # Allocate bits in order of significance
        for idx in sorted_indices:
            if remaining_bits <= 0:
                break
            
            # Allocate based on significance level
            significance_ratio = significance_values[idx] / np.max(significance_values)
            
            if significance_ratio > 0.8:
                bits_to_allocate = min(self.max_bits_per_band, remaining_bits)
            elif significance_ratio > 0.5:
                bits_to_allocate = min(self.max_bits_per_band // 2, remaining_bits)
            else:
                bits_to_allocate = min(self.min_bits_per_band, remaining_bits)
            
            allocation[idx] = bits_to_allocate
            remaining_bits -= bits_to_allocate
        
        return allocation
    
    def _optimal_allocation(self, 
                          significance_values: np.ndarray,
                          band_capabilities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimal allocation using dynamic programming approach.
        
        This method considers both significance and capacity constraints to find
        the optimal bit distribution.
        """
        n_bands = len(significance_values)
        
        if band_capabilities is None:
            band_capabilities = np.full(n_bands, self.max_bits_per_band)
        
        # Dynamic programming for optimal allocation
        # State: (band_index, remaining_bits) -> maximum utility
        dp = {}
        
        def dp_solve(band_idx: int, remaining_bits: int) -> Tuple[float, List[int]]:
            """Recursive DP solver."""
            if band_idx >= n_bands or remaining_bits <= 0:
                return 0.0, [0] * (n_bands - band_idx)
            
            if (band_idx, remaining_bits) in dp:
                return dp[(band_idx, remaining_bits)]
            
            best_utility = 0.0
            best_allocation = [0] * (n_bands - band_idx)
            
            # Try different bit allocations for current band
            max_bits_here = min(band_capabilities[band_idx], remaining_bits, self.max_bits_per_band)
            
            for bits in range(self.min_bits_per_band, max_bits_here + 1):
                # Utility from allocating 'bits' to current band
                current_utility = significance_values[band_idx] * np.sqrt(bits)  # Diminishing returns
                
                # Solve for remaining bands
                future_utility, future_allocation = dp_solve(band_idx + 1, remaining_bits - bits)
                
                total_utility = current_utility + future_utility
                
                if total_utility > best_utility:
                    best_utility = total_utility
                    best_allocation = [bits] + future_allocation
            
            dp[(band_idx, remaining_bits)] = (best_utility, best_allocation)
            return best_utility, best_allocation
        
        _, optimal_allocation = dp_solve(0, self.available_bits)
        return np.array(optimal_allocation[:n_bands], dtype=int)
    
    def _apply_constraints(self, allocation: np.ndarray, n_bands: int) -> np.ndarray:
        """Apply min/max constraints to bit allocation."""
        # Apply minimum constraint
        allocation = np.maximum(allocation, self.min_bits_per_band)
        
        # Apply maximum constraint
        allocation = np.minimum(allocation, self.max_bits_per_band)
        
        # Ensure we don't exceed total available bits
        total_allocated = np.sum(allocation)
        if total_allocated > self.available_bits:
            # Proportionally reduce allocations
            scale_factor = self.available_bits / total_allocated
            allocation = np.floor(allocation * scale_factor).astype(int)
            allocation = np.maximum(allocation, self.min_bits_per_band)
        
        return allocation
    
    def _compute_allocation_efficiency(self, 
                                    allocation: np.ndarray,
                                    significance_values: np.ndarray) -> Dict[str, float]:
        """Compute efficiency metrics for the allocation."""
        # Weighted utility per bit
        total_utility = np.sum(allocation * significance_values)
        total_bits = np.sum(allocation)
        
        utility_per_bit = total_utility / (total_bits + 1e-10)
        
        # Distribution uniformity (entropy-based)
        prob_dist = allocation / (np.sum(allocation) + 1e-10)
        entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
        max_entropy = np.log2(len(allocation))
        uniformity = entropy / max_entropy
        
        # Significance correlation
        correlation = np.corrcoef(allocation, significance_values)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        return {
            'utility_per_bit': utility_per_bit,
            'distribution_uniformity': uniformity,
            'significance_correlation': correlation,
            'efficiency_score': utility_per_bit * correlation * (1 + uniformity)
        }


class PerceptualAdaptiveBitAllocation(nn.Module):
    """
    Neural network module for end-to-end perceptual adaptive bit allocation.
    
    Integrates perceptual significance computation and bit allocation into the
    watermarking pipeline for joint optimization.
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 1000,
                 hop_length: int = 512,
                 n_critical_bands: int = 24,
                 total_bits: int = 1000,
                 learnable_allocation: bool = True):
        """
        Initialize the perceptual adaptive bit allocation module.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_critical_bands: Number of critical bands
            total_bits: Total watermark bits
            learnable_allocation: Whether to use learnable allocation parameters
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_critical_bands = n_critical_bands
        self.total_bits = total_bits
        
        # Initialize components
        self.significance_metric = PerceptualSignificanceMetric(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_critical_bands=n_critical_bands
        )
        
        self.bit_allocator = AdaptiveBitAllocator(
            total_bits=total_bits,
            allocation_strategy="optimal"
        )
        
        # Learnable parameters for allocation adaptation
        if learnable_allocation:
            self.significance_weights = nn.Parameter(torch.ones(n_critical_bands))
            self.allocation_bias = nn.Parameter(torch.zeros(n_critical_bands))
            self.adaptive_temperature = nn.Parameter(torch.tensor(1.0))
        else:
            self.significance_weights = None
            self.allocation_bias = None
            self.adaptive_temperature = None
    
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for adaptive bit allocation.
        
        Args:
            audio: Input audio tensor (batch_size, samples)
            
        Returns:
            Dictionary with allocation results
        """
        batch_size = audio.shape[0]
        device = audio.device
        
        batch_allocations = []
        batch_significance = []
        
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()
            
            # Compute perceptual significance
            significance_result = self.significance_metric.compute_band_significance(audio_np)
            significance = significance_result['band_significance']
            
            # Apply learnable weights if available
            if self.significance_weights is not None:
                significance_tensor = torch.from_numpy(significance).float().to(device)
                adapted_significance = significance_tensor * self.significance_weights
                adapted_significance = adapted_significance + self.allocation_bias
                
                # Apply temperature scaling
                adapted_significance = adapted_significance / self.adaptive_temperature
                
                # Convert back to numpy for allocation (detach to avoid gradient issues)
                significance = adapted_significance.detach().cpu().numpy()
            
            # Allocate bits
            allocation_result = self.bit_allocator.allocate_bits(significance)
            allocation = allocation_result['bit_allocation']
            
            batch_allocations.append(torch.from_numpy(allocation).float().to(device))
            batch_significance.append(torch.from_numpy(significance).float().to(device))
        
        # Stack batch results
        bit_allocations = torch.stack(batch_allocations)
        significance_values = torch.stack(batch_significance)
        
        return {
            'bit_allocations': bit_allocations,
            'significance_values': significance_values,
            'total_bits_per_sample': torch.sum(bit_allocations, dim=1),
            'allocation_entropy': self._compute_allocation_entropy(bit_allocations),
            'significance_weights': self.significance_weights,
            'allocation_bias': self.allocation_bias,
            'adaptive_temperature': self.adaptive_temperature
        }
    
    def _compute_allocation_entropy(self, allocations: torch.Tensor) -> torch.Tensor:
        """Compute entropy of bit allocation distribution."""
        # Normalize allocations to probabilities
        prob_allocations = allocations / (torch.sum(allocations, dim=1, keepdim=True) + 1e-10)
        
        # Compute entropy
        entropy = -torch.sum(prob_allocations * torch.log2(prob_allocations + 1e-10), dim=1)
        
        return entropy
    
    def get_allocation_for_bands(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get detailed allocation information for each critical band.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Detailed allocation information
        """
        result = self.forward(audio)
        
        # Add band-specific information
        band_info = {
            'band_centers': torch.from_numpy(self.significance_metric.band_centers).float(),
            'band_indices': torch.from_numpy(self.significance_metric.band_indices).long(),
            'frequency_ranges': self._compute_frequency_ranges()
        }
        
        result.update(band_info)
        return result
    
    def _compute_frequency_ranges(self) -> torch.Tensor:
        """Compute frequency ranges for each critical band."""
        frequencies = self.significance_metric.frequencies
        band_indices = self.significance_metric.band_indices
        
        ranges = []
        for band_idx in range(self.n_critical_bands):
            band_mask = band_indices == band_idx
            if np.any(band_mask):
                band_freqs = frequencies[band_mask]
                freq_range = [np.min(band_freqs), np.max(band_freqs)]
            else:
                freq_range = [0.0, 0.0]
            ranges.append(freq_range)
        
        return torch.tensor(ranges, dtype=torch.float32)


# Factory functions for easy instantiation
def create_perceptual_significance_metric(sample_rate: int = 16000,
                                        n_critical_bands: int = 24,
                                        method: str = "logarithmic") -> PerceptualSignificanceMetric:
    """
    Factory function to create a perceptual significance metric.
    
    Args:
        sample_rate: Audio sample rate
        n_critical_bands: Number of critical bands
        method: Significance computation method
        
    Returns:
        Configured PerceptualSignificanceMetric instance
    """
    return PerceptualSignificanceMetric(
        sample_rate=sample_rate,
        n_critical_bands=n_critical_bands,
        significance_method=method,
        temporal_averaging=True,
        adaptive_normalization=True
    )


def create_adaptive_bit_allocator(total_bits: int = 64,
                                strategy: str = "optimal") -> AdaptiveBitAllocator:
    """
    Factory function to create an adaptive bit allocator.
    
    Args:
        total_bits: Total number of watermark bits
        strategy: Allocation strategy
        
    Returns:
        Configured AdaptiveBitAllocator instance
    """
    return AdaptiveBitAllocator(
        total_bits=total_bits,
        allocation_strategy=strategy,
        min_bits_per_band=1,
        max_bits_per_band=min(8, total_bits // 4)
    )


if __name__ == "__main__":
    # Test the perceptual significance metric and adaptive bit allocation
    print("Testing Perceptual Significance Metric and Adaptive Bit Allocation...")
    
    # Generate test audio
    duration = 2.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create test signal with multiple frequency components
    test_audio = (0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
                 0.3 * np.sin(2 * np.pi * 880 * t) +   # A5 note  
                 0.2 * np.sin(2 * np.pi * 1760 * t) +  # A6 note
                 0.1 * np.random.randn(len(t)))        # Noise
    
    # Test significance metric
    significance_metric = create_perceptual_significance_metric(
        sample_rate=sample_rate,
        n_critical_bands=24,
        method="logarithmic"
    )
    
    significance_result = significance_metric.compute_band_significance(test_audio)
    print(f"Band significance shape: {significance_result['band_significance'].shape}")
    print(f"Total significance: {significance_result['total_significance']:.3f}")
    
    # Test bit allocator
    bit_allocator = create_adaptive_bit_allocator(total_bits=1000, strategy="optimal")
    allocation_result = bit_allocator.allocate_bits(significance_result['band_significance'])
    
    print(f"Bit allocation: {allocation_result['bit_allocation']}")
    print(f"Total allocated bits: {allocation_result['total_allocated']}")
    print(f"Allocation efficiency: {allocation_result['allocation_efficiency']['efficiency_score']:.3f}")
    
    # Test neural module
    neural_allocator = PerceptualAdaptiveBitAllocation(
        sample_rate=sample_rate,
        n_critical_bands=24,
        total_bits=1000,
        learnable_allocation=True
    )
    
    # Convert to tensor and test
    audio_tensor = torch.from_numpy(test_audio).unsqueeze(0).float()
    neural_result = neural_allocator(audio_tensor)
    
    print(f"Neural allocation shape: {neural_result['bit_allocations'].shape}")
    print(f"Allocation entropy: {neural_result['allocation_entropy'].item():.3f}")
    
    print("✅ Perceptual Significance Metric and Adaptive Bit Allocation test completed!")
    # ========================================================================================
    # Task 1.2.2.2: Advanced Dynamic Bit Allocation Algorithm
    # ========================================================================================
    
    def dynamic_allocate_bits(self, 
                            significance_values: np.ndarray,
                            audio_features: Optional[Dict[str, np.ndarray]] = None,
                            temporal_context: Optional[np.ndarray] = None,
                            band_capabilities: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Advanced dynamic bit allocation algorithm for Task 1.2.2.2.
        
        This algorithm implements a sophisticated multi-objective optimization approach
        that considers perceptual significance, temporal consistency, audio content
        characteristics, and dynamic constraints.
        
        Args:
            significance_values: Perceptual significance for each band
            audio_features: Additional audio characteristics (SNR, transient detection, etc.)
            temporal_context: Previous allocation states for temporal consistency
            band_capabilities: Dynamic capacity constraints per band
            
        Returns:
            Dictionary with comprehensive allocation results
        """
        n_bands = len(significance_values)
        
        # Step 1: Analyze audio content for context-aware allocation
        content_analysis = self._analyze_audio_content(significance_values, audio_features)
        
        # Step 2: Compute dynamic band capabilities based on content analysis
        dynamic_capabilities = self._compute_dynamic_capabilities(
            significance_values, content_analysis, band_capabilities
        )
        
        # Step 3: Multi-objective optimization for bit allocation
        initial_allocation = self._multi_objective_optimization(
            significance_values, dynamic_capabilities, content_analysis
        )
        
        # Step 4: Temporal consistency adjustment
        if temporal_context is not None:
            adjusted_allocation = self._apply_temporal_consistency(
                initial_allocation, temporal_context, significance_values
            )
        else:
            adjusted_allocation = initial_allocation
        
        # Step 5: Intelligent reallocation for optimization
        final_allocation = self._intelligent_reallocation(
            adjusted_allocation, significance_values, dynamic_capabilities
        )
        
        # Step 6: Validate and apply all constraints
        constrained_allocation = self._apply_dynamic_constraints(
            final_allocation, dynamic_capabilities, n_bands
        )
        
        # Step 7: Compute comprehensive metrics
        allocation_metrics = self._compute_advanced_metrics(
            constrained_allocation, significance_values, content_analysis
        )
        
        return {
            'bit_allocation': constrained_allocation,
            'dynamic_capabilities': dynamic_capabilities,
            'content_analysis': content_analysis,
            'allocation_metrics': allocation_metrics,
            'total_allocated': np.sum(constrained_allocation),
            'remaining_bits': self.available_bits - np.sum(constrained_allocation),
            'utilization_ratio': np.sum(constrained_allocation) / self.available_bits,
            'temporal_consistency_score': allocation_metrics.get('temporal_consistency', 0.0),
            'robustness_score': allocation_metrics.get('robustness_score', 0.0),
            'imperceptibility_score': allocation_metrics.get('imperceptibility_score', 0.0)
        }
    
    def _analyze_audio_content(self, 
                              significance_values: np.ndarray, 
                              audio_features: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        """
        Analyze audio content characteristics for context-aware allocation.
        
        Returns:
            Dictionary with content analysis metrics
        """
        analysis = {}
        
        # Significance distribution analysis
        sig_mean = np.mean(significance_values)
        sig_std = np.std(significance_values)
        sig_max = np.max(significance_values)
        sig_min = np.min(significance_values)
        
        analysis['significance_contrast'] = (sig_max - sig_min) / (sig_mean + 1e-10)
        analysis['significance_uniformity'] = 1.0 - (sig_std / (sig_mean + 1e-10))
        analysis['significance_entropy'] = self._compute_entropy(significance_values)
        
        # Frequency content analysis
        low_freq_significance = np.mean(significance_values[:6])    # Bands 0-5 (low freq)
        mid_freq_significance = np.mean(significance_values[6:18])  # Bands 6-17 (mid freq)
        high_freq_significance = np.mean(significance_values[18:])  # Bands 18+ (high freq)
        
        analysis['low_freq_dominance'] = low_freq_significance / (sig_mean + 1e-10)
        analysis['mid_freq_dominance'] = mid_freq_significance / (sig_mean + 1e-10)
        analysis['high_freq_dominance'] = high_freq_significance / (sig_mean + 1e-10)
        
        # Audio features integration
        if audio_features is not None:
            if 'snr_bands' in audio_features:
                analysis['average_snr'] = np.mean(audio_features['snr_bands'])
                analysis['snr_consistency'] = 1.0 - np.std(audio_features['snr_bands']) / (np.mean(audio_features['snr_bands']) + 1e-10)
            
            if 'transient_detection' in audio_features:
                analysis['transient_activity'] = np.mean(audio_features['transient_detection'])
            
            if 'spectral_stability' in audio_features:
                analysis['spectral_stability'] = audio_features['spectral_stability']
        
        # Content-based robustness prediction
        analysis['robustness_demand'] = self._predict_robustness_demand(significance_values, audio_features)
        
        return analysis
    
    def _compute_dynamic_capabilities(self, 
                                    significance_values: np.ndarray,
                                    content_analysis: Dict[str, float],
                                    base_capabilities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute dynamic band capabilities based on content analysis.
        
        Returns:
            Dynamic capacity constraints per band
        """
        n_bands = len(significance_values)
        
        if base_capabilities is None:
            base_capabilities = np.full(n_bands, self.max_bits_per_band)
        
        dynamic_caps = base_capabilities.copy().astype(float)
        
        # Adjust based on significance contrast
        contrast = content_analysis.get('significance_contrast', 1.0)
        if contrast > 2.0:  # High contrast - emphasize high-significance bands
            sig_normalized = significance_values / (np.max(significance_values) + 1e-10)
            dynamic_caps = dynamic_caps * (0.5 + 0.5 * sig_normalized)
        
        # Adjust based on frequency dominance
        low_dom = content_analysis.get('low_freq_dominance', 1.0)
        mid_dom = content_analysis.get('mid_freq_dominance', 1.0)
        high_dom = content_analysis.get('high_freq_dominance', 1.0)
        
        # Boost capacity for dominant frequency regions
        if low_dom > 1.2:
            dynamic_caps[:6] *= 1.2  # Boost low frequencies
        if mid_dom > 1.2:
            dynamic_caps[6:18] *= 1.3  # Boost mid frequencies (most important)
        if high_dom > 1.2:
            dynamic_caps[18:] *= 1.1  # Modest boost for high frequencies
        
        # Adjust based on robustness demand
        robustness_demand = content_analysis.get('robustness_demand', 1.0)
        if robustness_demand > 1.2:
            # High robustness demand - allow more bits per band
            dynamic_caps *= min(1.5, robustness_demand)
        
        # Ensure constraints are respected
        dynamic_caps = np.clip(dynamic_caps, self.min_bits_per_band, self.max_bits_per_band * 2)
        
        return dynamic_caps.astype(int)
    
    def _multi_objective_optimization(self, 
                                    significance_values: np.ndarray,
                                    capabilities: np.ndarray,
                                    content_analysis: Dict[str, float]) -> np.ndarray:
        """
        Multi-objective optimization for bit allocation considering multiple criteria.
        
        Optimizes for:
        1. Perceptual significance maximization
        2. Robustness (capacity utilization)
        3. Imperceptibility (balanced distribution)
        
        Returns:
            Optimized bit allocation
        """
        n_bands = len(significance_values)
        
        # Objective weights based on content analysis
        w_significance = 0.5  # Base weight for significance
        w_robustness = 0.3 + 0.2 * min(content_analysis.get('robustness_demand', 1.0), 2.0)
        w_imperceptibility = 0.2 + 0.1 * content_analysis.get('significance_uniformity', 0.5)
        
        # Normalize weights
        total_weight = w_significance + w_robustness + w_imperceptibility
        w_significance /= total_weight
        w_robustness /= total_weight
        w_imperceptibility /= total_weight
        
        # Particle Swarm Optimization for multi-objective optimization
        best_allocation = self._particle_swarm_allocation(
            significance_values, capabilities, 
            w_significance, w_robustness, w_imperceptibility
        )
        
        return best_allocation
    
    def _particle_swarm_allocation(self, 
                                 significance_values: np.ndarray,
                                 capabilities: np.ndarray,
                                 w_sig: float, w_rob: float, w_imp: float,
                                 n_particles: int = 20, n_iterations: int = 50) -> np.ndarray:
        """
        Particle Swarm Optimization for optimal bit allocation.
        
        Returns:
            Optimized allocation using PSO
        """
        n_bands = len(significance_values)
        
        # Initialize particles (allocation solutions)
        particles = []
        velocities = []
        personal_best = []
        personal_best_scores = []
        
        for _ in range(n_particles):
            # Initialize random allocation
            particle = self._generate_random_allocation(n_bands, capabilities)
            velocity = np.random.normal(0, 1, n_bands)
            
            particles.append(particle)
            velocities.append(velocity)
            personal_best.append(particle.copy())
            personal_best_scores.append(self._evaluate_allocation_fitness(
                particle, significance_values, w_sig, w_rob, w_imp
            ))
        
        # Find global best
        global_best_idx = np.argmax(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        # PSO parameters
        inertia = 0.9
        cognitive = 2.0
        social = 2.0
        
        # PSO iterations
        for iteration in range(n_iterations):
            for i in range(n_particles):
                # Update velocity
                r1, r2 = np.random.random(2)
                velocities[i] = (inertia * velocities[i] + 
                               cognitive * r1 * (personal_best[i] - particles[i]) +
                               social * r2 * (global_best - particles[i]))
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                particles[i] = self._constrain_allocation(particles[i], capabilities)
                
                # Evaluate fitness
                fitness = self._evaluate_allocation_fitness(
                    particles[i], significance_values, w_sig, w_rob, w_imp
                )
                
                # Update personal best
                if fitness > personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = fitness
                
                # Update global best
                if fitness > global_best_score:
                    global_best = particles[i].copy()
                    global_best_score = fitness
            
            # Decay inertia
            inertia *= 0.99
        
        return global_best.astype(int)
    
    def _generate_random_allocation(self, n_bands: int, capabilities: np.ndarray) -> np.ndarray:
        """Generate a random valid allocation."""
        allocation = np.zeros(n_bands)
        remaining_bits = self.available_bits
        
        for i in range(n_bands):
            if remaining_bits <= 0:
                allocation[i] = self.min_bits_per_band
            else:
                max_for_band = min(capabilities[i], remaining_bits, self.max_bits_per_band)
                allocation[i] = np.random.randint(self.min_bits_per_band, max_for_band + 1)
                remaining_bits -= allocation[i]
        
        return allocation
    
    def _constrain_allocation(self, allocation: np.ndarray, capabilities: np.ndarray) -> np.ndarray:
        """Constrain allocation to valid ranges."""
        # Clip to valid ranges
        allocation = np.clip(allocation, self.min_bits_per_band, capabilities)
        
        # Ensure total doesn't exceed available bits
        total = np.sum(allocation)
        if total > self.available_bits:
            scale_factor = self.available_bits / total
            allocation = allocation * scale_factor
            allocation = np.maximum(allocation, self.min_bits_per_band)
        
        return allocation
    
    def _evaluate_allocation_fitness(self, 
                                   allocation: np.ndarray,
                                   significance_values: np.ndarray,
                                   w_sig: float, w_rob: float, w_imp: float) -> float:
        """
        Evaluate fitness of an allocation for multi-objective optimization.
        
        Returns:
            Combined fitness score
        """
        # Significance objective: maximize weighted utility
        significance_utility = np.sum(allocation * significance_values)
        max_possible_utility = np.sum(self.max_bits_per_band * significance_values)
        significance_score = significance_utility / (max_possible_utility + 1e-10)
        
        # Robustness objective: maximize capacity utilization
        robustness_score = np.sum(allocation) / self.available_bits
        
        # Imperceptibility objective: minimize allocation variance (balanced distribution)
        normalized_allocation = allocation / (np.max(allocation) + 1e-10)
        allocation_entropy = -np.sum(normalized_allocation * np.log2(normalized_allocation + 1e-10))
        max_entropy = np.log2(len(allocation))
        imperceptibility_score = allocation_entropy / max_entropy
        
        # Combined fitness
        fitness = (w_sig * significance_score + 
                  w_rob * robustness_score + 
                  w_imp * imperceptibility_score)
        
        return fitness
    
    def _apply_temporal_consistency(self, 
                                  current_allocation: np.ndarray,
                                  previous_allocation: np.ndarray,
                                  significance_values: np.ndarray,
                                  consistency_weight: float = 0.3) -> np.ndarray:
        """
        Apply temporal consistency to avoid abrupt allocation changes.
        
        Returns:
            Temporally smoothed allocation
        """
        # Compute temporal consistency penalty for large changes
        allocation_change = np.abs(current_allocation - previous_allocation)
        max_change = np.max(self.max_bits_per_band - self.min_bits_per_band)
        
        # Smooth allocation using weighted average
        smoothed_allocation = ((1 - consistency_weight) * current_allocation + 
                              consistency_weight * previous_allocation)
        
        # Ensure smoothed allocation is still valid
        smoothed_allocation = self._constrain_allocation(
            smoothed_allocation, 
            np.full(len(significance_values), self.max_bits_per_band)
        )
        
        return smoothed_allocation.astype(int)
    
    def _intelligent_reallocation(self, 
                                allocation: np.ndarray,
                                significance_values: np.ndarray,
                                capabilities: np.ndarray) -> np.ndarray:
        """
        Intelligent reallocation to optimize bit usage.
        
        Uses iterative improvement to find better allocation distributions.
        
        Returns:
            Improved allocation
        """
        improved_allocation = allocation.copy()
        n_bands = len(allocation)
        
        # Iterative improvement
        for iteration in range(10):  # Maximum 10 improvement iterations
            improved = False
            
            for i in range(n_bands):
                for j in range(n_bands):
                    if i == j:
                        continue
                    
                    # Try transferring one bit from band i to band j
                    if (improved_allocation[i] > self.min_bits_per_band and 
                        improved_allocation[j] < capabilities[j]):
                        
                        # Calculate utility change
                        old_utility = (improved_allocation[i] * significance_values[i] + 
                                     improved_allocation[j] * significance_values[j])
                        
                        new_utility = ((improved_allocation[i] - 1) * significance_values[i] + 
                                     (improved_allocation[j] + 1) * significance_values[j])
                        
                        # If utility improves, make the transfer
                        if new_utility > old_utility:
                            improved_allocation[i] -= 1
                            improved_allocation[j] += 1
                            improved = True
            
            if not improved:
                break  # No more improvements possible
        
        return improved_allocation
    
    def _apply_dynamic_constraints(self, 
                                 allocation: np.ndarray,
                                 capabilities: np.ndarray,
                                 n_bands: int) -> np.ndarray:
        """
        Apply dynamic constraints ensuring allocation validity.
        
        Returns:
            Constrained allocation
        """
        constrained = allocation.copy()
        
        # Apply min/max constraints per band
        constrained = np.clip(constrained, self.min_bits_per_band, capabilities)
        
        # Ensure total doesn't exceed available bits
        total_allocated = np.sum(constrained)
        if total_allocated > self.available_bits:
            # Proportionally reduce allocations
            excess = total_allocated - self.available_bits
            
            # Reduce from bands with highest allocation first
            sorted_indices = np.argsort(constrained)[::-1]
            
            remaining_excess = excess
            for idx in sorted_indices:
                if remaining_excess <= 0:
                    break
                
                max_reduction = constrained[idx] - self.min_bits_per_band
                actual_reduction = min(max_reduction, remaining_excess)
                
                constrained[idx] -= actual_reduction
                remaining_excess -= actual_reduction
        
        return constrained.astype(int)
    
    def _compute_advanced_metrics(self, 
                                allocation: np.ndarray,
                                significance_values: np.ndarray,
                                content_analysis: Dict[str, float]) -> Dict[str, float]:
        """
        Compute comprehensive metrics for the allocation.
        
        Returns:
            Advanced metrics dictionary
        """
        metrics = {}
        
        # Basic efficiency metrics
        total_utility = np.sum(allocation * significance_values)
        max_possible_utility = np.sum(self.max_bits_per_band * significance_values)
        metrics['utility_efficiency'] = total_utility / (max_possible_utility + 1e-10)
        
        # Robustness metrics
        metrics['capacity_utilization'] = np.sum(allocation) / self.available_bits
        metrics['robustness_score'] = self._compute_robustness_score(allocation, significance_values)
        
        # Imperceptibility metrics
        metrics['imperceptibility_score'] = self._compute_imperceptibility_score(allocation)
        
        # Distribution analysis
        allocation_entropy = self._compute_entropy(allocation)
        max_entropy = np.log2(len(allocation))
        metrics['allocation_entropy'] = allocation_entropy
        metrics['distribution_uniformity'] = allocation_entropy / max_entropy
        
        # Significance correlation
        correlation = np.corrcoef(allocation, significance_values)[0, 1]
        metrics['significance_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        # Content-aware metrics
        metrics['content_alignment'] = self._compute_content_alignment(allocation, content_analysis)
        
        # Combined score
        metrics['overall_score'] = (0.4 * metrics['utility_efficiency'] + 
                                  0.3 * metrics['robustness_score'] + 
                                  0.2 * metrics['imperceptibility_score'] +
                                  0.1 * metrics['content_alignment'])
        
        return metrics
    
    def _predict_robustness_demand(self, 
                                 significance_values: np.ndarray,
                                 audio_features: Optional[Dict[str, np.ndarray]] = None) -> float:
        """
        Predict robustness demand based on audio characteristics.
        
        Returns:
            Robustness demand factor (1.0 = normal, >1.0 = high demand)
        """
        base_demand = 1.0
        
        # High significance variance suggests need for more robust encoding
        sig_variance = np.var(significance_values)
        if sig_variance > np.mean(significance_values):
            base_demand *= 1.2
        
        # Audio features contribution
        if audio_features is not None:
            if 'average_snr' in audio_features and audio_features['average_snr'] < 20:
                base_demand *= 1.3  # Low SNR needs more robustness
            
            if 'transient_activity' in audio_features and audio_features['transient_activity'] > 0.5:
                base_demand *= 1.15  # High transient activity needs robustness
        
        return base_demand
    
    def _compute_entropy(self, values: np.ndarray) -> float:
        """Compute entropy of a value distribution."""
        # Normalize to probabilities
        normalized = values / (np.sum(values) + 1e-10)
        normalized = normalized + 1e-10  # Avoid log(0)
        
        # Compute entropy
        entropy = -np.sum(normalized * np.log2(normalized))
        return entropy
    
    def _compute_robustness_score(self, allocation: np.ndarray, significance_values: np.ndarray) -> float:
        """Compute robustness score based on allocation and significance."""
        # Robustness is higher when high-significance bands get more bits
        weighted_allocation = allocation * significance_values
        total_weighted = np.sum(weighted_allocation)
        max_possible_weighted = np.sum(self.max_bits_per_band * significance_values)
        
        return total_weighted / (max_possible_weighted + 1e-10)
    
    def _compute_imperceptibility_score(self, allocation: np.ndarray) -> float:
        """Compute imperceptibility score (higher for more balanced distribution)."""
        # More balanced allocation is more imperceptible
        allocation_entropy = self._compute_entropy(allocation)
        max_entropy = np.log2(len(allocation))
        
        return allocation_entropy / max_entropy
    
    def _compute_content_alignment(self, allocation: np.ndarray, content_analysis: Dict[str, float]) -> float:
        """Compute how well allocation aligns with content characteristics."""
        alignment_score = 0.0
        
        # Check frequency dominance alignment
        low_freq_allocation = np.mean(allocation[:6])
        mid_freq_allocation = np.mean(allocation[6:18])
        high_freq_allocation = np.mean(allocation[18:])
        
        total_avg = np.mean(allocation)
        
        low_dom = content_analysis.get('low_freq_dominance', 1.0)
        mid_dom = content_analysis.get('mid_freq_dominance', 1.0)
        high_dom = content_analysis.get('high_freq_dominance', 1.0)
        
        # Score alignment between allocation and dominance
        if low_dom > 1.2 and low_freq_allocation > total_avg:
            alignment_score += 0.3
        if mid_dom > 1.2 and mid_freq_allocation > total_avg:
            alignment_score += 0.4  # Mid frequencies are most important
        if high_dom > 1.2 and high_freq_allocation > total_avg:
            alignment_score += 0.3
        
        return min(1.0, alignment_score)
