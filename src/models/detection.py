"""Face detection using InsightFace RetinaFace."""
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src.core.config import settings

logger = logging.getLogger(__name__)


class DetectedFace:
    """Container for detected face data."""

    def __init__(
        self,
        bbox: np.ndarray,
        landmarks: np.ndarray,
        score: float,
        image: np.ndarray,
        insightface_face: Optional[object] = None
    ):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.landmarks = landmarks  # 5 points: eyes, nose, mouth corners
        self.score = score  # Detection confidence
        self.image = image  # Cropped face image
        self.insightface_face = insightface_face  # Original InsightFace face object

    @property
    def width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    def __repr__(self) -> str:
        return f"DetectedFace(bbox={self.bbox}, score={self.score:.3f}, size={self.width}x{self.height})"


class FaceDetector:
    """Face detection using InsightFace RetinaFace."""

    def __init__(self, use_gpu: bool = True, enable_preprocessing: bool = True):
        """Initialize detector.

        Args:
            use_gpu: Whether to use GPU acceleration
            enable_preprocessing: Enable image preprocessing for long-range detection
        """
        self.use_gpu = use_gpu
        self.enable_preprocessing = enable_preprocessing
        self.model: Optional[FaceAnalysis] = None
        self._initialized = False

        logger.info(f"FaceDetector initialized (GPU: {use_gpu}, Preprocessing: {enable_preprocessing})")

    def _lazy_load(self) -> None:
        """Lazy load model on first use."""
        if self._initialized:
            return

        logger.info("Loading detection model pack '%s'...", settings.detection_model)

        providers = ["CUDAExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]

        self.model = FaceAnalysis(
            name=settings.detection_model,
            providers=providers,
            allowed_modules=['detection'],
            root=str(settings.model_cache_dir)
        )

        self.model.prepare(
            ctx_id=0 if self.use_gpu else -1,
            det_size=settings.det_size,
            det_thresh=settings.det_thresh,
        )

        self._initialized = True
        logger.info("Detection model '%s' loaded successfully", settings.detection_model)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect faces in image.

        Args:
            image: BGR image as numpy array (H, W, 3)

        Returns:
            List of DetectedFace objects
        """
        self._lazy_load()

        if image is None or image.size == 0:
            logger.warning("Empty image provided to detect()")
            return []

        # Apply preprocessing for long-range detection if enabled
        processed_image = image
        if self.enable_preprocessing:
            try:
                from src.preprocessing import preprocess_pipeline
                processed_image = preprocess_pipeline(image, preset="balanced")
                logger.debug("Applied preprocessing for long-range detection")
            except Exception as e:
                logger.warning(f"Preprocessing failed, using original image: {e}")
                processed_image = image

        # Run detection
        faces = self.model.get(processed_image)

        results = []
        for face in faces:
            # Extract face region
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            face_img = image[y1:y2, x1:x2].copy()

            detected = DetectedFace(
                bbox=bbox,
                landmarks=face.kps,
                score=float(face.det_score),
                image=face_img,
                insightface_face=face
            )
            results.append(detected)

        logger.debug(f"Detected {len(results)} faces")
        return results

    def detect_from_file(self, image_path: Path) -> list[DetectedFace]:
        """Detect faces from image file.

        Args:
            image_path: Path to image file

        Returns:
            List of DetectedFace objects
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        return self.detect(image)
