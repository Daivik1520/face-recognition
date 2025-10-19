# Facial Data Augmentation System

## Overview

The facial data augmentation system enhances the robustness and accuracy of the face recognition model by generating multiple augmented versions of enrollment images. This enables the model to reliably detect and recognize faces under various real-world conditions including different lighting, angles, distances, and quality scenarios.

## Features

### Augmentation Types

1. **Lighting Variations**
   - Simulates different lighting conditions (dim light, bright light, indoor/outdoor)
   - Adjusts brightness and contrast
   - Helps model handle varying illumination scenarios

2. **Cropping and Scaling**
   - Generates zoomed-in and zoomed-out versions
   - Creates off-center crops
   - Simulates faces at different distances and partially in frame

3. **Rotation and Orientation**
   - Applies small rotations (-15° to +15°)
   - Includes horizontal flips
   - Accounts for head tilts and orientation changes

4. **Blur Variations**
   - Adds Gaussian blur
   - Simulates motion blur
   - Handles lower-quality camera conditions

5. **Noise Variations**
   - Adds Gaussian noise at different levels
   - Simulates sensor noise in low-light conditions
   - Improves robustness to image quality variations

## Configuration

### Augmentation Presets

Three preset configurations are available:

#### 1. Minimal
- **Use case**: Quick enrollment, limited computational resources
- **Augmentations**: 2 lighting + 2 crops + 2 rotations
- **Total images**: ~7 per original image
- **No blur or noise variations**

```python
{
    "num_lighting": 2,
    "num_crops": 2,
    "num_rotations": 2,
    "add_blur": False,
    "add_noise": False,
}
```

#### 2. Balanced (Default)
- **Use case**: Standard enrollment, good balance of quality and speed
- **Augmentations**: 4 lighting + 3 crops + 3 rotations + blur + noise
- **Total images**: ~15 per original image
- **Recommended for most scenarios**

```python
{
    "num_lighting": 4,
    "num_crops": 3,
    "num_rotations": 3,
    "add_blur": True,
    "add_noise": True,
}
```

#### 3. Aggressive
- **Use case**: Maximum robustness, challenging environments
- **Augmentations**: 6 lighting + 5 crops + 5 rotations + blur + noise
- **Total images**: ~20+ per original image
- **Best for critical applications**

```python
{
    "num_lighting": 6,
    "num_crops": 5,
    "num_rotations": 5,
    "add_blur": True,
    "add_noise": True,
}
```

## Usage

### API Endpoint

The `/enroll` endpoint now supports augmentation parameters:

```bash
POST /api/enroll
```

**Parameters:**
- `name` (required): Person's name
- `files` (required): One or more image files
- `use_augmentation` (optional, default: `true`): Enable/disable augmentation
- `augmentation_preset` (optional, default: `"balanced"`): Preset to use (`"minimal"`, `"balanced"`, or `"aggressive"`)

**Example using curl:**

```bash
# Enroll with default balanced augmentation
curl -X POST "http://localhost:8000/api/enroll" \
  -F "name=John Doe" \
  -F "files=@photo.jpg" \
  -F "use_augmentation=true" \
  -F "augmentation_preset=balanced"

# Enroll with aggressive augmentation
curl -X POST "http://localhost:8000/api/enroll" \
  -F "name=Jane Smith" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg" \
  -F "use_augmentation=true" \
  -F "augmentation_preset=aggressive"

# Enroll without augmentation
curl -X POST "http://localhost:8000/api/enroll" \
  -F "name=Bob Wilson" \
  -F "files=@photo.jpg" \
  -F "use_augmentation=false"
```

### Response Format

**With Augmentation:**
```json
{
  "message": "Successfully enrolled John Doe with augmentation",
  "total_enrolled": 512,
  "images_processed": 1,
  "successful_enrollments": 15,
  "total_embeddings": 20,
  "augmented_count": 15,
  "original_count": 1,
  "avg_quality": 0.847,
  "augmentation_used": true,
  "augmentation_preset": "balanced"
}
```

**Without Augmentation:**
```json
{
  "message": "Successfully enrolled Bob Wilson",
  "total_enrolled": 1,
  "images_processed": 1,
  "augmentation_used": false
}
```

## Python API

### Single Image Enrollment

```python
from src.face_system import FaceRecognitionSystem
import cv2

# Initialize system with augmentation enabled
face_system = FaceRecognitionSystem(
    enable_augmentation=True,
    augmentation_preset="balanced"
)

# Load image
image = cv2.imread("person.jpg")

# Enroll with augmentation
result = face_system.enroll_with_augmentation("John Doe", image)

print(f"Enrolled {result['enrolled_count']} images")
print(f"Generated {result['augmented_count']} augmented versions")
print(f"Total embeddings: {result['total_embeddings']}")
print(f"Average quality: {result['avg_quality']:.3f}")
```

### Multiple Image Enrollment

```python
# Load multiple images
images = [
    cv2.imread("photo1.jpg"),
    cv2.imread("photo2.jpg"),
    cv2.imread("photo3.jpg")
]

# Enroll with augmentation on each image
result = face_system.enroll_multiple_with_augmentation("Jane Smith", images)

print(f"Processed {result['original_count']} original images")
print(f"Generated {result['augmented_count']} total augmented versions")
print(f"Successfully enrolled {result['enrolled_count']} embeddings")
```

### Custom Augmentation Configuration

```python
from src.augmentation import AugmentationConfig

# Get a preset and customize it
config = AugmentationConfig.get_preset("balanced")
config["num_lighting"] = 6  # Increase lighting variations
config["add_blur"] = False  # Disable blur

# Use custom config
result = face_system.enroll_with_augmentation(
    "Custom Person",
    image,
    augmentation_config=config
)
```

## Storage

### Augmented Images

When `save_augmented=True` (default), augmented images are saved to:
```
data/augmented/{person_name}/
  ├── aug_000.jpg  (original)
  ├── aug_001.jpg  (lighting variation 1)
  ├── aug_002.jpg  (lighting variation 2)
  ├── aug_003.jpg  (crop variation 1)
  └── ...
```

This allows for:
- Visual inspection of augmented data
- Debugging augmentation quality
- Manual review of training data

### Embeddings Database

All embeddings (from original and augmented images) are stored in:
```
data/processed/face_embeddings.json
```

Each embedding includes:
- 512-dimensional face vector
- Quality score
- Detection confidence score

The system automatically keeps only the best embeddings per person (default: top 20).

## Benefits

### Improved Recognition Accuracy

- **Lighting Robustness**: Recognizes faces in varying lighting conditions
- **Pose Invariance**: Handles different head angles and orientations
- **Distance Flexibility**: Works with faces at different distances
- **Quality Tolerance**: More resilient to blur and noise

### Real-World Performance

- **Indoor/Outdoor**: Adapts to different environments
- **Camera Quality**: Works with various camera qualities
- **Partial Occlusion**: Better handles partially visible faces
- **Motion Handling**: Improved performance with motion blur

## Best Practices

### 1. Choose the Right Preset

- **Minimal**: Testing, development, resource-constrained environments
- **Balanced**: Production deployments, general use cases
- **Aggressive**: High-security applications, challenging conditions

### 2. Multiple Original Images

For best results, provide 2-5 high-quality original images:
- Different angles (front, slight left, slight right)
- Different expressions (neutral, smiling)
- Different lighting conditions

### 3. Quality Over Quantity

- Start with high-quality original images
- Ensure faces are clearly visible and well-lit
- Avoid heavily blurred or occluded faces

### 4. Monitor Performance

Check the response metrics:
- `avg_quality`: Should be > 0.5 for good enrollment
- `successful_enrollments`: Higher is better
- `total_embeddings`: Indicates diversity of training data

### 5. Incremental Enrollment

You can enroll the same person multiple times to add more embeddings:
```python
# Initial enrollment
face_system.enroll_with_augmentation("John", image1)

# Add more data later
face_system.enroll_with_augmentation("John", image2)
```

The system keeps the best embeddings up to the configured limit.

## Performance Considerations

### Processing Time

Approximate enrollment times (single image, CPU):

| Preset     | Augmented Images | Processing Time |
|------------|------------------|-----------------|
| Minimal    | ~7               | 2-3 seconds     |
| Balanced   | ~15              | 4-6 seconds     |
| Aggressive | ~20+             | 6-10 seconds    |

### Storage Requirements

Per person (balanced preset):
- Embeddings: ~40 KB (20 embeddings × 2 KB each)
- Augmented images (optional): ~1-2 MB (15 images × 100 KB each)

### Memory Usage

- Augmentation process: ~50-100 MB per image
- Embedding extraction: ~200-300 MB (InsightFace model)

## Troubleshooting

### Low Quality Scores

**Problem**: `avg_quality` < 0.3

**Solutions**:
- Use higher quality original images
- Ensure good lighting in original photos
- Check that faces are clearly visible
- Try the "minimal" preset first

### Too Few Successful Enrollments

**Problem**: `successful_enrollments` much lower than `augmented_count`

**Solutions**:
- Original image may have quality issues
- Face might be too small or poorly lit
- Try adjusting `min_embedding_quality` threshold
- Provide multiple original images

### Slow Processing

**Problem**: Enrollment takes too long

**Solutions**:
- Use "minimal" preset
- Reduce number of original images
- Disable augmented image saving
- Consider GPU acceleration

### Storage Issues

**Problem**: Too much disk space used

**Solutions**:
- Disable augmented image saving: `save_augmented=False`
- Reduce `max_embeddings_per_person`
- Use "minimal" preset
- Periodically clean old augmented images

## Technical Details

### Augmentation Pipeline

1. **Original Image** → Always included first
2. **Lighting Variations** → Brightness/contrast adjustments
3. **Crop Variations** → Zoom in/out, off-center crops
4. **Rotation Variations** → Small rotations, flips
5. **Blur Variations** → Gaussian blur, motion blur
6. **Noise Variations** → Gaussian noise at different levels

### Quality Filtering

Each augmented image is processed through:
1. Face detection (det_thresh=0.6)
2. Quality assessment (size, pose, sharpness)
3. Embedding extraction (512-d ArcFace)
4. Quality filtering (min_quality=0.2)
5. Best embedding selection (top 20)

### Embedding Management

- Embeddings sorted by quality × detection score
- Only best N embeddings kept per person
- Automatic deduplication of similar embeddings
- Persistent storage in JSON format

## Future Enhancements

Potential improvements:
- Color space variations (HSV adjustments)
- Advanced pose augmentation (3D transformations)
- Occlusion simulation (glasses, masks)
- Age progression/regression
- Makeup and accessory variations
- Weather condition simulation (fog, rain effects)

## References

- InsightFace ArcFace: https://github.com/deepinsight/insightface
- Data Augmentation for Face Recognition: Various academic papers
- OpenCV Image Processing: https://docs.opencv.org/
