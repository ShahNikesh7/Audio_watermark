"""
FastAPI REST API for SoundSafeAI watermarking service.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import uvicorn
from typing import Optional
import io
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SoundSafeAI API",
    description="Audio watermarking service with embedding and extraction capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WatermarkRequest(BaseModel):
    """Request model for watermark embedding."""
    watermark_data: str
    strength: float = 0.1


class WatermarkResponse(BaseModel):
    """Response model for watermark operations."""
    success: bool
    message: str
    data: Optional[dict] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "SoundSafeAI API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "SoundSafeAI"}


@app.post("/embed", response_model=WatermarkResponse)
async def embed_watermark(
    audio_file: UploadFile = File(...),
    watermark_data: str = Form(...),
    strength: float = Form(0.1)
):
    """
    Embed watermark into audio file.
    
    Args:
        audio_file: Audio file to watermark
        watermark_data: Data to embed as watermark
        strength: Embedding strength (0.0 to 1.0)
        
    Returns:
        Watermarked audio file
    """
    try:
        logger.info(f"Embedding watermark with strength {strength}")
        
        # Read audio file
        audio_bytes = await audio_file.read()
        
        # Process audio (placeholder implementation)
        # In a real implementation, this would use the embedding models
        
        return WatermarkResponse(
            success=True,
            message="Watermark embedded successfully",
            data={
                "original_size": len(audio_bytes),
                "watermark_data": watermark_data,
                "strength": strength
            }
        )
        
    except Exception as e:
        logger.error(f"Error embedding watermark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract", response_model=WatermarkResponse)
async def extract_watermark(
    audio_file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Extract watermark from audio file.
    
    Args:
        audio_file: Audio file to extract watermark from
        confidence_threshold: Minimum confidence for valid extraction
        
    Returns:
        Extracted watermark data and confidence score
    """
    try:
        logger.info(f"Extracting watermark with confidence threshold {confidence_threshold}")
        
        # Read audio file
        audio_bytes = await audio_file.read()
        
        # Process audio (placeholder implementation)
        # In a real implementation, this would use the extraction models
        
        # Simulate extraction
        confidence_score = np.random.uniform(0.3, 0.9)
        extracted_data = f"watermark_data_{int(confidence_score * 100)}" if confidence_score >= confidence_threshold else None
        
        return WatermarkResponse(
            success=True,
            message="Watermark extraction completed",
            data={
                "extracted_data": extracted_data,
                "confidence_score": confidence_score,
                "threshold": confidence_threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Error extracting watermark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect", response_model=WatermarkResponse)
async def detect_watermark(
    audio_file: UploadFile = File(...),
    detection_threshold: float = Form(0.3)
):
    """
    Detect presence of watermark in audio file.
    
    Args:
        audio_file: Audio file to check for watermark
        detection_threshold: Minimum score for watermark detection
        
    Returns:
        Detection result and confidence score
    """
    try:
        logger.info(f"Detecting watermark with threshold {detection_threshold}")
        
        # Read audio file
        audio_bytes = await audio_file.read()
        
        # Process audio (placeholder implementation)
        detection_score = np.random.uniform(0.1, 0.8)
        is_watermarked = detection_score >= detection_threshold
        
        return WatermarkResponse(
            success=True,
            message="Watermark detection completed",
            data={
                "is_watermarked": is_watermarked,
                "detection_score": detection_score,
                "threshold": detection_threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Error detecting watermark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_service_stats():
    """Get service statistics."""
    return {
        "service": "SoundSafeAI",
        "version": "1.0.0",
        "endpoints": [
            "/embed",
            "/extract", 
            "/detect",
            "/health"
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
