"""
Image preprocessing module for long-range face detection enhancement.
Improves image quality before detection to enable better recognition at distance.
"""
import cv2
import numpy as np
from typing import Optional


def preprocess_for_long_range(
    image: np.ndarray,
    upscale_factor: float = 2.0,
    enable_clahe: bool = True,
    enable_denoise: bool = True,
    denoise_strength: int = 7
) -> np.ndarray:
    """
    Preprocess image to enhance distant face detection.
    
    Args:
        image: Input BGR image
        upscale_factor: Factor to upscale image (default 2.0 for 2x)
        enable_clahe: Apply CLAHE contrast enhancement
        enable_denoise: Apply denoising
        denoise_strength: Denoising filter strength (3-21, higher = more smoothing)
    
    Returns:
        Enhanced BGR image
    """
    if image is None or image.size == 0:
        return image
    
    # Step 1: Upscale using bicubic interpolation
    if upscale_factor != 1.0:
        new_width = int(image.shape[1] * upscale_factor)
        new_height = int(image.shape[0] * upscale_factor)
        image = cv2.resize(
            image, 
            (new_width, new_height), 
            interpolation=cv2.INTER_CUBIC
        )
    
    # Step 2: CLAHE contrast enhancement
    if enable_clahe:
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Merge back
        lab = cv2.merge([l_channel, a_channel, b_channel])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Step 3: Denoise
    if enable_denoise:
        image = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=denoise_strength,
            hColor=denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
    
    return image


def adaptive_sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply adaptive sharpening to enhance edges.
    
    Args:
        image: Input BGR image
        strength: Sharpening strength (0.5-2.0)
    
    Returns:
        Sharpened image
    """
    if image is None or image.size == 0:
        return image
    
    # Create sharpening kernel
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ]) * strength / 9.0
    
    # Apply kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened


def enhance_brightness(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Enhance brightness using gamma correction.
    
    Args:
        image: Input BGR image
        gamma: Gamma value (>1 brightens, <1 darkens)
    
    Returns:
        Brightness-adjusted image
    """
    if image is None or image.size == 0:
        return image
    
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")
    
    # Apply gamma correction
    return cv2.LUT(image, table)


def preprocess_pipeline(
    image: np.ndarray,
    preset: str = "balanced"
) -> np.ndarray:
    """
    Apply full preprocessing pipeline with preset configurations.
    
    Args:
        image: Input BGR image
        preset: One of "fast", "balanced", "quality"
            - fast: Minimal processing, fastest
            - balanced: Good quality/speed trade-off (default)
            - quality: Maximum quality, slowest
    
    Returns:
        Fully preprocessed image
    """
    if preset == "fast":
        # Just upscale
        return preprocess_for_long_range(
            image,
            upscale_factor=1.5,
            enable_clahe=False,
            enable_denoise=False
        )
    
    elif preset == "balanced":
        # Upscale + CLAHE
        enhanced = preprocess_for_long_range(
            image,
            upscale_factor=2.0,
            enable_clahe=True,
            enable_denoise=False
        )
        return enhanced
    
    elif preset == "quality":
        # Full pipeline
        enhanced = preprocess_for_long_range(
            image,
            upscale_factor=2.0,
            enable_clahe=True,
            enable_denoise=True,
            denoise_strength=5
        )
        # Add sharpening
        enhanced = adaptive_sharpen(enhanced, strength=0.8)
        return enhanced
    
    else:
        raise ValueError(f"Unknown preset: {preset}. Use 'fast', 'balanced', or 'quality'")
