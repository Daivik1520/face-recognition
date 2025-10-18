# Face Recognition Range Enhancement Log

**Goal**: Increase detection range from 6 feet to 20-30 feet while preserving accuracy and functionality

**Date Started**: October 18, 2025, 8:03 PM IST

---

## Phase 1: Quick Wins Implementation (Target: 12-15ft range)

### Current Baseline Configuration
- **Detection Resolution**: 640x640
- **Detection Threshold**: 0.35
- **Similarity Threshold**: 0.45
- **Min Face Size**: 40px
- **Model**: buffalo_l
- **Hardware**: CPU only
- **Current Range**: ~6 feet

### Changes Log

#### Change 1: Increase Detection Resolution
**Status**: In Progress
**Target**: 1280x1280 (2x increase)
**Expected Impact**: +100% range improvement
**Trade-off**: ~75% slower processing
**Files Modified**: 
- `.env`
- `src/face_system.py`

#### Change 2: Lower Detection Thresholds
**Status**: Pending
**Target**: 
- DET_THRESH: 0.35 → 0.20
- SIMILARITY_THRESHOLD: 0.45 → 0.38
- MIN_FACE_SIZE: 40 → 25
**Expected Impact**: +50% range improvement
**Trade-off**: Potential for more false positives
**Files Modified**: `.env`

#### Change 3: Image Preprocessing Pipeline
**Status**: Pending
**Components**:
- Upscaling (2x bicubic interpolation)
- CLAHE contrast enhancement
- Denoising
**Expected Impact**: +30% quality improvement for distant faces
**Trade-off**: Additional processing time
**Files Modified**: 
- `src/models/detection.py` or new `src/preprocessing.py`

#### Change 4: Adaptive Confidence Thresholds
**Status**: Pending
**Logic**: Adjust threshold based on detected face size
**Expected Impact**: Better balance between range and accuracy
**Files Modified**: `src/face_system.py`

---

## Testing & Validation

### Test Distances
- [ ] 6 feet (baseline)
- [ ] 10 feet
- [ ] 15 feet
- [ ] 20 feet
- [ ] 25 feet
- [ ] 30 feet

### Metrics to Track
- Detection rate (% faces detected)
- Recognition accuracy (% correct matches)
- False positive rate
- Processing time per frame
- Confidence scores at each distance

---

## Implementation Timeline

**Start**: 8:03 PM IST, Oct 18, 2025
**Estimated Completion**: 8:30 PM IST, Oct 18, 2025

---

## Notes & Observations

*This section will be updated as implementation progresses...*

---

## Implementation Progress

### ✅ Change 1: Detection Resolution Increase
**Time**: 8:05 PM IST
**Status**: COMPLETED
**Details**:
- Updated `.env`: DET_SIZE_WIDTH=1280, DET_SIZE_HEIGHT=1280
- Modified `src/core/config.py`: Added det_size_width/height fields with @property for backward compatibility
- Previous: 640x640 (409,600 pixels)
- New: 1280x1280 (1,638,400 pixels) - 4x pixel count
**Expected Result**: Should detect faces 2x farther away

### ✅ Change 2: Threshold Adjustments
**Time**: 8:05 PM IST
**Status**: COMPLETED
**Details**:
- DET_THRESH: 0.35 → 0.20 (detect weaker faces)
- SIMILARITY_THRESHOLD: 0.45 → 0.38 (accept lower confidence matches)
- MIN_FACE_SIZE: 40 → 25 pixels (allow smaller faces)
**Expected Result**: +50% range, may increase false positives


### ✅ Change 3: Image Preprocessing Pipeline
**Time**: 8:08 PM IST
**Status**: COMPLETED
**Details**:
- Created new module: `src/preprocessing.py`
- Implemented preprocessing pipeline with 3 presets:
  - **fast**: 1.5x upscale only
  - **balanced**: 2x upscale + CLAHE contrast (default)
  - **quality**: Full pipeline with denoising + sharpening
- Modified `src/models/detection.py`:
  - Added `enable_preprocessing` parameter to FaceDetector
  - Integrated preprocessing before detection
  - Uses "balanced" preset by default
**Pipeline Steps**:
1. Bicubic upscaling (2x) - increases resolution
2. CLAHE contrast enhancement - improves visibility
3. Optional denoising - reduces noise artifacts
4. Optional sharpening - enhances edges
**Expected Result**: +30-40% quality improvement for distant faces

