"""
Demo script to visualize facial data augmentation.
This script shows examples of each augmentation type.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.augmentation import FaceAugmentation, AugmentationConfig


def create_demo_grid(images, titles, grid_cols=5):
    """Create a grid visualization of images."""
    if not images:
        return None
    
    # Resize all images to same size for grid
    target_size = (200, 200)
    resized_images = []
    
    for img in images:
        if img is not None and img.size > 0:
            resized = cv2.resize(img, target_size)
            # Add title text
            resized_with_text = resized.copy()
            resized_images.append(resized_with_text)
    
    if not resized_images:
        return None
    
    # Calculate grid dimensions
    n_images = len(resized_images)
    n_rows = (n_images + grid_cols - 1) // grid_cols
    
    # Create grid
    grid_rows = []
    for row_idx in range(n_rows):
        start_idx = row_idx * grid_cols
        end_idx = min(start_idx + grid_cols, n_images)
        row_images = resized_images[start_idx:end_idx]
        
        # Pad row if needed
        while len(row_images) < grid_cols:
            row_images.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
        
        # Add titles
        for i, (img, title) in enumerate(zip(row_images, titles[start_idx:end_idx])):
            cv2.putText(img, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        row = np.hstack(row_images)
        grid_rows.append(row)
    
    grid = np.vstack(grid_rows)
    return grid


def demo_lighting_variations():
    """Demonstrate lighting augmentation."""
    print("\n" + "="*60)
    print("DEMO: Lighting Variations")
    print("="*60)
    
    # Create a simple test image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    cv2.circle(img, (200, 200), 100, (180, 150, 120), -1)
    cv2.circle(img, (170, 180), 15, (50, 50, 50), -1)
    cv2.circle(img, (230, 180), 15, (50, 50, 50), -1)
    cv2.ellipse(img, (200, 240), (40, 20), 0, 0, 180, (100, 50, 50), 2)
    
    augmentor = FaceAugmentation(save_augmented=False)
    
    # Generate lighting variations
    variations = augmentor._generate_lighting_variations(img, count=6)
    
    all_images = [img] + variations
    titles = ["Original", "Dark 1", "Dark 2", "Bright 1", "Bright 2", 
              "High Contrast", "Low Contrast"]
    
    print(f"Generated {len(variations)} lighting variations")
    print("Simulates: dim light, bright light, indoor/outdoor conditions")
    
    # Save grid
    grid = create_demo_grid(all_images[:7], titles[:7], grid_cols=4)
    if grid is not None:
        output_path = "data/demo_lighting.jpg"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, grid)
        print(f"✓ Saved visualization to: {output_path}")
    
    return all_images


def demo_crop_variations():
    """Demonstrate cropping augmentation."""
    print("\n" + "="*60)
    print("DEMO: Crop & Scale Variations")
    print("="*60)
    
    # Create test image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    cv2.circle(img, (200, 200), 100, (180, 150, 120), -1)
    cv2.circle(img, (170, 180), 15, (50, 50, 50), -1)
    cv2.circle(img, (230, 180), 15, (50, 50, 50), -1)
    cv2.ellipse(img, (200, 240), (40, 20), 0, 0, 180, (100, 50, 50), 2)
    
    augmentor = FaceAugmentation(save_augmented=False)
    
    # Generate crop variations
    variations = augmentor._generate_crop_variations(img, count=5)
    
    all_images = [img] + variations
    titles = ["Original", "Zoom In 85%", "Zoom In 90%", "Zoom Out 110%", 
              "Zoom Out 115%", "Off-Center"]
    
    print(f"Generated {len(variations)} crop variations")
    print("Simulates: different distances, partially visible faces")
    
    # Save grid
    grid = create_demo_grid(all_images[:6], titles[:6], grid_cols=3)
    if grid is not None:
        output_path = "data/demo_crops.jpg"
        cv2.imwrite(output_path, grid)
        print(f"✓ Saved visualization to: {output_path}")
    
    return all_images


def demo_rotation_variations():
    """Demonstrate rotation augmentation."""
    print("\n" + "="*60)
    print("DEMO: Rotation & Orientation Variations")
    print("="*60)
    
    # Create test image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    cv2.circle(img, (200, 200), 100, (180, 150, 120), -1)
    cv2.circle(img, (170, 180), 15, (50, 50, 50), -1)
    cv2.circle(img, (230, 180), 15, (50, 50, 50), -1)
    cv2.ellipse(img, (200, 240), (40, 20), 0, 0, 180, (100, 50, 50), 2)
    
    augmentor = FaceAugmentation(save_augmented=False)
    
    # Generate rotation variations
    variations = augmentor._generate_rotation_variations(img, count=6)
    
    all_images = [img] + variations
    titles = ["Original", "Rotate -15°", "Rotate -8°", "Rotate -3°", 
              "Rotate +3°", "Rotate +8°", "Flipped"]
    
    print(f"Generated {len(variations)} rotation variations")
    print("Simulates: head tilts, orientation changes")
    
    # Save grid
    grid = create_demo_grid(all_images[:7], titles[:7], grid_cols=4)
    if grid is not None:
        output_path = "data/demo_rotations.jpg"
        cv2.imwrite(output_path, grid)
        print(f"✓ Saved visualization to: {output_path}")
    
    return all_images


def demo_blur_variations():
    """Demonstrate blur augmentation."""
    print("\n" + "="*60)
    print("DEMO: Blur Variations")
    print("="*60)
    
    # Create test image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    cv2.circle(img, (200, 200), 100, (180, 150, 120), -1)
    cv2.circle(img, (170, 180), 15, (50, 50, 50), -1)
    cv2.circle(img, (230, 180), 15, (50, 50, 50), -1)
    cv2.ellipse(img, (200, 240), (40, 20), 0, 0, 180, (100, 50, 50), 2)
    
    augmentor = FaceAugmentation(save_augmented=False)
    
    # Generate blur variations
    variations = augmentor._generate_blur_variations(img, count=3)
    
    all_images = [img] + variations
    titles = ["Original", "Gaussian Blur 1", "Gaussian Blur 2", "Motion Blur"]
    
    print(f"Generated {len(variations)} blur variations")
    print("Simulates: camera quality issues, motion blur")
    
    # Save grid
    grid = create_demo_grid(all_images[:4], titles[:4], grid_cols=4)
    if grid is not None:
        output_path = "data/demo_blur.jpg"
        cv2.imwrite(output_path, grid)
        print(f"✓ Saved visualization to: {output_path}")
    
    return all_images


def demo_noise_variations():
    """Demonstrate noise augmentation."""
    print("\n" + "="*60)
    print("DEMO: Noise Variations")
    print("="*60)
    
    # Create test image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    cv2.circle(img, (200, 200), 100, (180, 150, 120), -1)
    cv2.circle(img, (170, 180), 15, (50, 50, 50), -1)
    cv2.circle(img, (230, 180), 15, (50, 50, 50), -1)
    cv2.ellipse(img, (200, 240), (40, 20), 0, 0, 180, (100, 50, 50), 2)
    
    augmentor = FaceAugmentation(save_augmented=False)
    
    # Generate noise variations
    variations = augmentor._generate_noise_variations(img, count=2)
    
    all_images = [img] + variations
    titles = ["Original", "Low Noise", "High Noise"]
    
    print(f"Generated {len(variations)} noise variations")
    print("Simulates: sensor noise in low-light conditions")
    
    # Save grid
    grid = create_demo_grid(all_images[:3], titles[:3], grid_cols=3)
    if grid is not None:
        output_path = "data/demo_noise.jpg"
        cv2.imwrite(output_path, grid)
        print(f"✓ Saved visualization to: {output_path}")
    
    return all_images


def demo_full_augmentation():
    """Demonstrate full augmentation pipeline."""
    print("\n" + "="*60)
    print("DEMO: Full Augmentation Pipeline")
    print("="*60)
    
    # Create test image
    img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    cv2.circle(img, (200, 200), 100, (180, 150, 120), -1)
    cv2.circle(img, (170, 180), 15, (50, 50, 50), -1)
    cv2.circle(img, (230, 180), 15, (50, 50, 50), -1)
    cv2.ellipse(img, (200, 240), (40, 20), 0, 0, 180, (100, 50, 50), 2)
    
    augmentor = FaceAugmentation(save_augmented=False)
    
    # Test each preset
    for preset_name in ["minimal", "balanced", "aggressive"]:
        print(f"\n--- {preset_name.upper()} Preset ---")
        config = AugmentationConfig.get_preset(preset_name)
        
        augmented = augmentor.augment_for_enrollment(
            img,
            f"demo_{preset_name}",
            **config
        )
        
        print(f"Generated {len(augmented)} total images")
        print(f"  Original: 1")
        print(f"  Augmented: {len(augmented) - 1}")
        
        # Save first 12 images as grid
        sample_images = augmented[:12]
        titles = [f"Image {i+1}" for i in range(len(sample_images))]
        titles[0] = "Original"
        
        grid = create_demo_grid(sample_images, titles, grid_cols=4)
        if grid is not None:
            output_path = f"data/demo_full_{preset_name}.jpg"
            cv2.imwrite(output_path, grid)
            print(f"✓ Saved sample grid to: {output_path}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("FACIAL DATA AUGMENTATION - VISUAL DEMO")
    print("="*60)
    print("\nThis demo generates visual examples of each augmentation type.")
    print("Output images will be saved to the 'data/' directory.")
    
    try:
        # Run individual demos
        demo_lighting_variations()
        demo_crop_variations()
        demo_rotation_variations()
        demo_blur_variations()
        demo_noise_variations()
        
        # Run full pipeline demo
        demo_full_augmentation()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60)
        print("\nGenerated visualization files:")
        print("  - data/demo_lighting.jpg")
        print("  - data/demo_crops.jpg")
        print("  - data/demo_rotations.jpg")
        print("  - data/demo_blur.jpg")
        print("  - data/demo_noise.jpg")
        print("  - data/demo_full_minimal.jpg")
        print("  - data/demo_full_balanced.jpg")
        print("  - data/demo_full_aggressive.jpg")
        print("\n✓ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
