"""
BCH Error Correction Codes for CNN-based Audio Watermarking
Implements Bose-Chaudhuri-Hocquenghem (BCH) codes for robust watermark encoding/decoding
specifically optimized for CNN perceptual models.

Task 1.2.1: Error Correction Codes Implementation for CNN branch
- 1.2.1.1: Implement BCH codes using reedsolo library
- 1.2.1.2: Determine optimal BCH parameters based on robustness requirements
- 1.2.1.3: Integrate BCH encoder/decoder into CNN watermark processes
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
import warnings

# Try to import reedsolo, with fallback if not available
try:
    import reedsolo
    REEDSOLO_AVAILABLE = True
except ImportError:
    REEDSOLO_AVAILABLE = False
    warnings.warn("reedsolo library not available. Using simple redundancy encoding as fallback.")

logger = logging.getLogger(__name__)


class CNNBCHEncoder(nn.Module):
    """
    BCH Encoder optimized for CNN-based watermark embedding.
    
    Implements Bose-Chaudhuri-Hocquenghem codes to add error correction
    capability to watermark bits before CNN-based embedding.
    """
    
    def __init__(self,
                 message_length: int = 1000,
                 code_rate: float = 0.5,
                 primitive_poly: int = 0x11d,  # Default primitive polynomial
                 fcr: int = 1,  # First consecutive root
                 prim: int = 1,  # Primitive element
                 use_optimized_params: bool = True,
                 cnn_awareness: bool = True):
        """
        Initialize BCH encoder with CNN-optimized parameters.
        
        Args:
            message_length: Length of original message bits
            code_rate: Code rate (message_length / codeword_length)
            primitive_poly: Primitive polynomial for GF operations
            fcr: First consecutive root
            prim: Primitive element
            use_optimized_params: Whether to use optimized parameters
            cnn_awareness: Enable CNN-specific optimizations
        """
        super(CNNBCHEncoder, self).__init__()
        
        self.message_length = message_length
        self.code_rate = code_rate
        self.primitive_poly = primitive_poly
        self.fcr = fcr
        self.prim = prim
        self.use_optimized_params = use_optimized_params
        self.cnn_awareness = cnn_awareness
        
        # Calculate codeword length based on practical Reed-Solomon parameters
        # Use 8-bit symbols (bytes) for Reed-Solomon processing
        message_bytes = (message_length + 7) // 8  # Convert bits to bytes, round up

        # Use standard Reed-Solomon parameters that work reliably
        if message_bytes <= 64:
            # Smaller byte payloads: use higher protection
            rs_parity_bytes = 16
            rs_total_bytes = message_bytes + rs_parity_bytes
        elif message_bytes <= 128:
            # Medium byte payloads: balanced protection
            rs_parity_bytes = 32
            rs_total_bytes = message_bytes + rs_parity_bytes
        else:
            # Large byte payloads: use chunked approach
            self.chunk_size = 64  # Process in 64-byte chunks
            self.num_chunks = (message_bytes + self.chunk_size - 1) // self.chunk_size

            # Each chunk gets error correction
            chunk_parity_bytes = 16  # 16 parity bytes per chunk
            chunk_total_bytes = self.chunk_size + chunk_parity_bytes

            rs_total_bytes = chunk_total_bytes * self.num_chunks
            rs_parity_bytes = rs_total_bytes - message_bytes

        # Convert back to bit-level parameters for compatibility
        self.codeword_length = rs_total_bytes * 8  # Convert bytes back to bits
        self.parity_length = rs_parity_bytes * 8   # Convert to bits

        # Store RS-specific parameters for proper codec initialization
        self.rs_message_bytes = message_bytes
        self.rs_parity_bytes = rs_parity_bytes
        self.rs_total_bytes = rs_total_bytes
        
        # Initialize BCH parameters
        self._initialize_bch_params()
        
        # CNN-specific parameters
        if self.cnn_awareness:
            self.spatial_weight = nn.Parameter(torch.ones(1))
            self.frequency_weight = nn.Parameter(torch.ones(1))
        
        logger.info(f"Initialized CNN BCH Encoder: message_len={message_length}, "
                   f"codeword_len={self.codeword_length}, rate={code_rate:.3f}")
    
    def _initialize_bch_params(self):
        """Initialize BCH codec parameters."""
        if REEDSOLO_AVAILABLE:
            try:
                # Use the RS-specific parameters calculated above
                if hasattr(self, 'rs_parity_bytes'):
                    # Use the byte-level parameters for proper RS operation
                    self.rs_codec = reedsolo.RSCodec(
                        nsym=self.rs_parity_bytes,
                        nsize=self.rs_total_bytes,
                        fcr=self.fcr,
                        prim=self.prim,
                        generator=self.primitive_poly,
                        c_exp=8
                    )
                    logger.info(f"BCH codec initialized with RS parameters: "
                              f"message_bytes={self.rs_message_bytes}, "
                              f"parity_bytes={self.rs_parity_bytes}, "
                              f"total_bytes={self.rs_total_bytes}")
                elif hasattr(self, 'chunk_size'):
                    # Legacy chunked approach for very large messages
                    chunk_message_bytes = min(self.chunk_size, self.rs_message_bytes)
                    chunk_parity_bytes = min(16, self.rs_parity_bytes // max(1, self.num_chunks))
                    chunk_total_bytes = chunk_message_bytes + chunk_parity_bytes

                    self.rs_codec = reedsolo.RSCodec(
                        nsym=chunk_parity_bytes,
                        nsize=chunk_total_bytes,
                        fcr=self.fcr,
                        prim=self.prim,
                        generator=self.primitive_poly,
                        c_exp=8
                    )
                    logger.info(f"BCH codec initialized with chunked RS: "
                              f"chunk_bytes={chunk_message_bytes}, parity={chunk_parity_bytes}")
                else:
                    # Standard approach for smaller messages - convert to bytes
                    message_bytes = (self.message_length + 7) // 8
                    parity_bytes = min(32, (self.codeword_length - self.message_length) // 8)
                    total_bytes = message_bytes + parity_bytes

                    self.rs_codec = reedsolo.RSCodec(
                        nsym=parity_bytes,
                        nsize=total_bytes,
                        fcr=self.fcr,
                        prim=self.prim,
                        generator=self.primitive_poly,
                        c_exp=8
                    )
                    logger.info(f"BCH codec initialized: nsym={parity_bytes}, nsize={total_bytes}")

                self.encoder_type = "reedsolo"
                logger.info("BCH encoder initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize reedsolo: {e}")
                logger.warning("This may be due to parameter constraints. Consider adjusting message_length or code_rate.")
                self._initialize_fallback_encoder()
        else:
            self._initialize_fallback_encoder()
    
    def _initialize_fallback_encoder(self):
        """Initialize fallback encoder when reedsolo is not available."""
        self.encoder_type = "fallback"
        # Simple repetition code parameters
        self.repetition_factor = max(2, self.parity_length // self.message_length + 1)
        logger.info(f"Using fallback repetition encoder (factor={self.repetition_factor})")
    
    def forward(self, message_bits: torch.Tensor) -> torch.Tensor:
        """
        Encode message bits with BCH error correction.
        
        Args:
            message_bits: Input message bits [batch_size, message_length]
            
        Returns:
            Encoded codeword bits [batch_size, codeword_length]
        """
        batch_size = message_bits.shape[0]
        
        if self.encoder_type == "reedsolo":
            return self._encode_reedsolo(message_bits)
        else:
            return self._encode_fallback(message_bits)
    
    def _encode_reedsolo(self, message_bits: torch.Tensor) -> torch.Tensor:
        """Encode using reedsolo library."""
        batch_size = message_bits.shape[0]
        encoded_batch = []
        
        for i in range(batch_size):
            # Convert bits to bytes
            bits = message_bits[i].cpu().numpy().astype(np.uint8)
            
            # Pad to byte boundary if necessary
            padded_bits = self._pad_to_bytes(bits)
            bytes_data = np.packbits(padded_bits)
            
            try:
                # Encode with Reed-Solomon
                encoded_bytes = self.rs_codec.encode(bytes_data)
                
                # Convert back to bits
                encoded_bits = np.unpackbits(encoded_bytes)[:self.codeword_length]
                
                # Ensure correct length
                if len(encoded_bits) < self.codeword_length:
                    encoded_bits = np.pad(encoded_bits, 
                                        (0, self.codeword_length - len(encoded_bits)))
                elif len(encoded_bits) > self.codeword_length:
                    encoded_bits = encoded_bits[:self.codeword_length]
                    
                encoded_batch.append(torch.tensor(encoded_bits, dtype=torch.float32))
                
            except Exception as e:
                logger.warning(f"BCH encoding failed for sample {i}: {e}")
                # Fallback to repetition
                encoded_bits = self._repetition_encode(bits)
                encoded_batch.append(torch.tensor(encoded_bits, dtype=torch.float32))
        
        return torch.stack(encoded_batch).to(message_bits.device)
    
    def _encode_fallback(self, message_bits: torch.Tensor) -> torch.Tensor:
        """Encode using fallback repetition method."""
        batch_size = message_bits.shape[0]
        encoded_batch = []
        
        for i in range(batch_size):
            bits = message_bits[i].cpu().numpy().astype(np.uint8)
            encoded_bits = self._repetition_encode(bits)
            encoded_batch.append(torch.tensor(encoded_bits, dtype=torch.float32))
        
        return torch.stack(encoded_batch).to(message_bits.device)
    
    def _repetition_encode(self, bits: np.ndarray) -> np.ndarray:
        """Simple repetition encoding."""
        # Repeat each bit for redundancy
        repeated = np.repeat(bits, self.repetition_factor)
        
        # Truncate or pad to desired codeword length
        if len(repeated) >= self.codeword_length:
            return repeated[:self.codeword_length]
        else:
            return np.pad(repeated, (0, self.codeword_length - len(repeated)))
    
    def _pad_to_bytes(self, bits: np.ndarray) -> np.ndarray:
        """Pad bit array to byte boundary."""
        remainder = len(bits) % 8
        if remainder != 0:
            padding = 8 - remainder
            return np.pad(bits, (0, padding))
        return bits


class CNNBCHDecoder(nn.Module):
    """
    BCH Decoder optimized for CNN-based watermark extraction.
    
    Implements BCH decoding with error correction capability
    for watermark bits extracted by CNN models.
    """
    
    def __init__(self,
                 message_length: int = 1000,
                 code_rate: float = 0.5,
                 primitive_poly: int = 0x11d,
                 fcr: int = 1,
                 prim: int = 1,
                 cnn_awareness: bool = True):
        """
        Initialize BCH decoder with CNN-optimized parameters.
        
        Args:
            message_length: Length of original message bits
            code_rate: Code rate (message_length / codeword_length)
            primitive_poly: Primitive polynomial for GF operations
            fcr: First consecutive root
            prim: Primitive element
            cnn_awareness: Enable CNN-specific optimizations
        """
        super(CNNBCHDecoder, self).__init__()
        
        self.message_length = message_length
        self.code_rate = code_rate
        self.primitive_poly = primitive_poly
        self.fcr = fcr
        self.prim = prim
        self.cnn_awareness = cnn_awareness
        
        # Calculate codeword length
        self.codeword_length = int(message_length / code_rate)
        self.parity_length = self.codeword_length - message_length
        
        # Initialize BCH parameters
        self._initialize_bch_params()
        
        # CNN-specific parameters
        if self.cnn_awareness:
            self.confidence_threshold = nn.Parameter(torch.tensor(0.5))
        
        logger.info(f"Initialized CNN BCH Decoder: message_len={message_length}, "
                   f"codeword_len={self.codeword_length}, rate={code_rate:.3f}")
    
    def _initialize_bch_params(self):
        """Initialize BCH codec parameters."""
        if REEDSOLO_AVAILABLE:
            try:
                # Create Reed-Solomon codec for BCH simulation
                self.rs_codec = reedsolo.RSCodec(
                    nsym=self.parity_length,
                    nsize=self.codeword_length,
                    fcr=self.fcr,
                    prim=self.prim,
                    generator=self.primitive_poly,
                    c_exp=8
                )
                self.decoder_type = "reedsolo"
                logger.info("Using reedsolo BCH decoder")
            except Exception as e:
                logger.warning(f"Failed to initialize reedsolo: {e}")
                self._initialize_fallback_decoder()
        else:
            self._initialize_fallback_decoder()
    
    def _initialize_fallback_decoder(self):
        """Initialize fallback decoder when reedsolo is not available."""
        self.decoder_type = "fallback"
        # Simple majority vote parameters
        self.repetition_factor = max(2, self.parity_length // self.message_length + 1)
        logger.info(f"Using fallback majority vote decoder (factor={self.repetition_factor})")
    
    def forward(self, received_bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode received bits with BCH error correction.
        
        Args:
            received_bits: Received codeword bits [batch_size, codeword_length]
            
        Returns:
            Tuple of (decoded_message_bits [batch_size, message_length], 
                     error_indicators [batch_size])
        """
        if self.decoder_type == "reedsolo":
            return self._decode_reedsolo(received_bits)
        else:
            return self._decode_fallback(received_bits)
    
    def _decode_reedsolo(self, received_bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode using reedsolo library."""
        batch_size = received_bits.shape[0]
        decoded_batch = []
        error_indicators = []
        
        for i in range(batch_size):
            # Convert bits to bytes
            bits = received_bits[i].cpu().numpy()
            bits_uint8 = (bits > 0.5).astype(np.uint8)
            
            # Pad to byte boundary
            padded_bits = self._pad_to_bytes(bits_uint8)
            bytes_data = np.packbits(padded_bits)
            
            try:
                # Decode with Reed-Solomon
                decoded_bytes, decoded_msg, errata_pos = self.rs_codec.decode(bytes_data, return_errors=True)
                
                # Convert back to bits
                decoded_bits = np.unpackbits(decoded_bytes)[:self.message_length]
                
                # Ensure correct length
                if len(decoded_bits) < self.message_length:
                    decoded_bits = np.pad(decoded_bits, 
                                        (0, self.message_length - len(decoded_bits)))
                elif len(decoded_bits) > self.message_length:
                    decoded_bits = decoded_bits[:self.message_length]
                
                decoded_batch.append(torch.tensor(decoded_bits, dtype=torch.float32))
                error_indicators.append(torch.tensor(len(errata_pos), dtype=torch.float32))
                
            except Exception as e:
                logger.warning(f"BCH decoding failed for sample {i}: {e}")
                # Fallback to majority vote
                decoded_bits, error_count = self._majority_vote_decode(bits_uint8)
                decoded_batch.append(torch.tensor(decoded_bits, dtype=torch.float32))
                error_indicators.append(torch.tensor(error_count, dtype=torch.float32))
        
        decoded_tensor = torch.stack(decoded_batch).to(received_bits.device)
        error_tensor = torch.stack(error_indicators).to(received_bits.device)
        
        return decoded_tensor, error_tensor
    
    def _decode_fallback(self, received_bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode using fallback majority vote method."""
        batch_size = received_bits.shape[0]
        decoded_batch = []
        error_indicators = []
        
        for i in range(batch_size):
            bits = received_bits[i].cpu().numpy()
            bits_uint8 = (bits > 0.5).astype(np.uint8)
            decoded_bits, error_count = self._majority_vote_decode(bits_uint8)
            
            decoded_batch.append(torch.tensor(decoded_bits, dtype=torch.float32))
            error_indicators.append(torch.tensor(error_count, dtype=torch.float32))
        
        decoded_tensor = torch.stack(decoded_batch).to(received_bits.device)
        error_tensor = torch.stack(error_indicators).to(received_bits.device)
        
        return decoded_tensor, error_tensor
    
    def _majority_vote_decode(self, bits: np.ndarray) -> Tuple[np.ndarray, float]:
        """Majority vote decoding for repetition codes."""
        decoded_bits = []
        error_count = 0
        
        # Process in groups based on repetition factor
        for i in range(0, min(len(bits), self.message_length * self.repetition_factor), 
                      self.repetition_factor):
            group = bits[i:i + self.repetition_factor]
            if len(group) == 0:
                break
                
            # Majority vote
            ones = np.sum(group)
            decoded_bit = 1 if ones > len(group) / 2 else 0
            decoded_bits.append(decoded_bit)
            
            # Count disagreements as errors
            if ones != len(group) and ones != 0:
                error_count += min(ones, len(group) - ones)
        
        # Pad or truncate to message length
        decoded_array = np.array(decoded_bits)
        if len(decoded_array) < self.message_length:
            decoded_array = np.pad(decoded_array, 
                                 (0, self.message_length - len(decoded_array)))
        elif len(decoded_array) > self.message_length:
            decoded_array = decoded_array[:self.message_length]
        
        return decoded_array, error_count
    
    def _pad_to_bytes(self, bits: np.ndarray) -> np.ndarray:
        """Pad bit array to byte boundary."""
        remainder = len(bits) % 8
        if remainder != 0:
            padding = 8 - remainder
            return np.pad(bits, (0, padding))
        return bits


class CNNBCHWatermarkProtector(nn.Module):
    """
    Complete BCH-based watermark protection system for CNN models.
    
    Combines encoding, embedding-specific optimization, and decoding
    for robust watermark protection in CNN-based audio watermarking.
    """
    
    def __init__(self,
                 message_length: int = 1000,
                 robustness_level: str = "medium",
                 cnn_model_type: str = "mobilenetv3"):
        """
        Initialize BCH watermark protector for CNN models.
        
        Args:
            message_length: Length of watermark message in bits
            robustness_level: "low", "medium", "high" - determines code rate
            cnn_model_type: Type of CNN model for optimization
        """
        super(CNNBCHWatermarkProtector, self).__init__()
        
        self.message_length = message_length
        self.robustness_level = robustness_level
        self.cnn_model_type = cnn_model_type
        
        # Determine optimal parameters based on robustness level
        self.code_rate, self.params = self._get_optimal_parameters(robustness_level)
        
        # Initialize encoder and decoder
        self.encoder = CNNBCHEncoder(
            message_length=message_length,
            code_rate=self.code_rate,
            **self.params
        )
        
        self.decoder = CNNBCHDecoder(
            message_length=message_length,
            code_rate=self.code_rate,
            primitive_poly=self.params.get("primitive_poly", 0x11d),
            fcr=self.params.get("fcr", 1),
            prim=self.params.get("prim", 1),
            cnn_awareness=self.params.get("cnn_awareness", True)
        )
        
        # CNN-specific optimizations
        self._initialize_cnn_optimizations()
        
        logger.info(f"Initialized CNN BCH Watermark Protector: "
                   f"msg_len={message_length}, robustness={robustness_level}, "
                   f"rate={self.code_rate:.3f}, model={cnn_model_type}")
    
    def _get_optimal_parameters(self, robustness_level: str) -> Tuple[float, Dict[str, Any]]:
        """
        Get optimal BCH parameters based on robustness requirements.
        
        Args:
            robustness_level: Required robustness level
            
        Returns:
            Tuple of (code_rate, parameters_dict)
        """
        if robustness_level == "low":
            # High rate, low error correction
            return 0.75, {
                "primitive_poly": 0x11d,
                "fcr": 1,
                "prim": 1,
                "use_optimized_params": True,
                "cnn_awareness": True
            }
        elif robustness_level == "medium":
            # Balanced rate and error correction
            return 0.5, {
                "primitive_poly": 0x11d,
                "fcr": 1,
                "prim": 1,
                "use_optimized_params": True,
                "cnn_awareness": True
            }
        elif robustness_level == "high":
            # Low rate, high error correction
            return 0.33, {
                "primitive_poly": 0x11d,
                "fcr": 1,
                "prim": 1,
                "use_optimized_params": True,
                "cnn_awareness": True
            }
        else:
            raise ValueError(f"Unknown robustness level: {robustness_level}")
    
    def _initialize_cnn_optimizations(self):
        """Initialize CNN-specific optimizations."""
        if self.cnn_model_type == "mobilenetv3":
            # MobileNetV3-specific parameters
            self.spatial_adaptation = nn.Parameter(torch.ones(1))
            self.channel_adaptation = nn.Parameter(torch.ones(1))
        elif self.cnn_model_type == "efficientnet":
            # EfficientNet-specific parameters
            self.compound_scaling = nn.Parameter(torch.ones(1))
            self.efficiency_weight = nn.Parameter(torch.ones(1))
        else:
            # Generic CNN parameters
            self.generic_weight = nn.Parameter(torch.ones(1))
    
    def encode_watermark(self, message_bits: torch.Tensor) -> torch.Tensor:
        """
        Encode watermark message with BCH error correction.
        
        Args:
            message_bits: Original watermark bits [batch_size, message_length]
            
        Returns:
            BCH-encoded codeword [batch_size, codeword_length]
        """
        return self.encoder(message_bits)
    
    def decode_watermark(self, received_bits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode received bits and recover original watermark.
        
        Args:
            received_bits: Received codeword bits [batch_size, codeword_length]
            
        Returns:
            Tuple of (decoded_message [batch_size, message_length], 
                     quality_metrics dict)
        """
        decoded_message, error_indicators = self.decoder(received_bits)
        
        # Calculate quality metrics
        quality_metrics = {
            "error_count": error_indicators,
            "error_rate": error_indicators / self.encoder.codeword_length,
            "reliability": torch.exp(-error_indicators),  # Exponential reliability
            "code_rate": torch.tensor(self.code_rate),
            "effective_capacity": torch.tensor(self.message_length / self.encoder.codeword_length)
        }
        
        return decoded_message, quality_metrics
    
    def evaluate_robustness(self, 
                           original_message: torch.Tensor,
                           noise_levels: List[float]) -> Dict[str, torch.Tensor]:
        """
        Evaluate BCH performance under different noise conditions.
        
        Args:
            original_message: Original message bits [batch_size, message_length]
            noise_levels: List of noise levels to test
            
        Returns:
            Dictionary of robustness metrics
        """
        results = {
            "noise_levels": torch.tensor(noise_levels),
            "bit_error_rates": [],
            "decode_success_rates": [],
            "quality_scores": []
        }
        
        # Encode the message
        encoded = self.encode_watermark(original_message)
        
        for noise_level in noise_levels:
            # Add noise to encoded message
            noise = torch.randn_like(encoded) * noise_level
            noisy_received = encoded + noise
            
            # Decode and evaluate
            decoded, metrics = self.decode_watermark(noisy_received)
            
            # Calculate bit error rate
            bit_errors = torch.sum(torch.abs(decoded - original_message), dim=1)
            ber = bit_errors / self.message_length
            
            # Calculate success rate (perfect decode)
            success_rate = torch.mean((bit_errors == 0).float())
            
            # Calculate quality score
            quality_score = torch.mean(metrics["reliability"])
            
            results["bit_error_rates"].append(torch.mean(ber))
            results["decode_success_rates"].append(success_rate)
            results["quality_scores"].append(quality_score)
        
        # Convert lists to tensors
        results["bit_error_rates"] = torch.stack(results["bit_error_rates"])
        results["decode_success_rates"] = torch.stack(results["decode_success_rates"])
        results["quality_scores"] = torch.stack(results["quality_scores"])
        
        return results
    
    def get_codeword_length(self) -> int:
        """Get the length of BCH codewords."""
        return self.encoder.codeword_length
    
    def get_message_length(self) -> int:
        """Get the length of original messages."""
        return self.message_length
    
    def get_code_rate(self) -> float:
        """Get the BCH code rate."""
        return self.code_rate


def optimize_bch_parameters_for_cnn(cnn_model_type: str,
                                   robustness_requirements: Dict[str, float],
                                   capacity_constraints: Dict[str, int]) -> Dict[str, Any]:
    """
    Optimize BCH parameters specifically for CNN-based watermarking.
    
    Args:
        cnn_model_type: Type of CNN model ("mobilenetv3", "efficientnet", etc.)
        robustness_requirements: Dict with BER tolerance, SNR requirements, etc.
        capacity_constraints: Dict with max_codeword_length, min_message_length, etc.
        
    Returns:
        Dictionary of optimal BCH parameters
    """
    logger.info(f"Optimizing BCH parameters for {cnn_model_type}")
    
    # Extract requirements
    target_ber = robustness_requirements.get("target_ber", 1e-4)
    min_snr = robustness_requirements.get("min_snr_db", 10.0)
    max_codeword_len = capacity_constraints.get("max_codeword_length", 256)
    min_message_len = capacity_constraints.get("min_message_length", 32)
    
    # CNN-specific adjustments
    if cnn_model_type == "mobilenetv3":
        # MobileNetV3 works well with shorter codewords due to efficiency
        preferred_codeword_len = min(128, max_codeword_len)
        code_rate_factor = 1.1  # Slightly higher rate
    elif cnn_model_type == "efficientnet":
        # EfficientNet can handle longer codewords efficiently
        preferred_codeword_len = max_codeword_len
        code_rate_factor = 1.0  # Standard rate
    else:
        # Generic CNN
        preferred_codeword_len = max_codeword_len // 2
        code_rate_factor = 0.9  # Slightly lower rate for safety
    
    # Calculate optimal code rate based on SNR requirements
    # Higher SNR allows higher code rates
    snr_linear = 10 ** (min_snr / 10.0)
    base_code_rate = min(0.75, max(0.25, 0.5 + 0.1 * np.log10(snr_linear)))
    optimal_code_rate = base_code_rate * code_rate_factor
    
    # Calculate message length
    optimal_message_len = max(min_message_len, 
                            int(preferred_codeword_len * optimal_code_rate))
    
    # Ensure feasible parameters
    if optimal_message_len > preferred_codeword_len:
        optimal_message_len = preferred_codeword_len // 2
        optimal_code_rate = optimal_message_len / preferred_codeword_len
    
    optimal_params = {
        "message_length": optimal_message_len,
        "code_rate": optimal_code_rate,
        "codeword_length": int(optimal_message_len / optimal_code_rate),
        "primitive_poly": 0x11d,
        "fcr": 1,
        "prim": 1,
        "use_optimized_params": True,
        "cnn_awareness": True,
        "robustness_level": "medium" if optimal_code_rate > 0.4 else "high"
    }
    
    logger.info(f"Optimal BCH parameters: rate={optimal_code_rate:.3f}, "
               f"msg_len={optimal_message_len}, codeword_len={optimal_params['codeword_length']}")
    
    return optimal_params
