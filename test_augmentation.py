"""
Test script for facial data augmentation system.
Run this to verify the augmentation implementation.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.augmentation import FaceAugmentation, AugmentationConfig
from src.face_system import FaceRecognitionSystem


def create_test_image():
    """Create a simple test image with a colored rectangle (simulating a face)."""
    # Create a 640x480 BGR image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Draw a "face" (rectangle with features)
    cv2.rectangle(image, (200, 100), (440, 380), (200, 180, 150), -1)
    
    # Eyes
    cv2.circle(image, (270, 200), 20, (50, 50, 50), -1)
    cv2.circle(image, (370, 200), 20, (50, 50, 50), -1)
    
    # Nose
    cv2.line(image, (320, 220), (320, 280), (150, 130, 120), 3)
    
    # Mouth
    cv2.ellipse(image, (320, 320), (40, 20), 0, 0, 180, (100, 50, 50), 2)
    
    return image


def test_augmentation_module():
    """Test the augmentation module independently."""
    print("=" * 60)
    print("TEST 1: Augmentation Module")
    print("=" * 60)
    
    # Create test image
    test_image = create_test_image()
    print(f"✓ Created test image: {test_image.shape}")
    
    # Initialize augmentor
    augmentor = FaceAugmentation(save_augmented=True, output_dir="data/test_augmented")
    print("✓ Initialized FaceAugmentation")
    
    # Test each preset
    for preset_name in ["minimal", "balanced", "aggressive"]:
        print(f"\n--- Testing '{preset_name}' preset ---")
        config = AugmentationConfig.get_preset(preset_name)
        print(f"Config: {config}")
        
        augmented = augmentor.augment_for_enrollment(
            test_image,
            f"test_person_{preset_name}",
            **config
        )
        
        print(f"✓ Generated {len(augmented)} augmented images")
        print(f"  - Original: 1")
        print(f"  - Augmented: {len(augmented) - 1}")
    
    print("\n✓ Augmentation module test PASSED")
    return True


def test_face_system_integration():
    """Test integration with FaceRecognitionSystem."""
    print("\n" + "=" * 60)
    print("TEST 2: Face System Integration")
    print("=" * 60)
    
    try:
        # Initialize face system with augmentation
        print("Initializing FaceRecognitionSystem...")
        face_system = FaceRecognitionSystem(
            enable_augmentation=True,
            augmentation_preset="minimal"  # Use minimal for faster testing
        )
        print("✓ FaceRecognitionSystem initialized")
        
        # Check augmentation is enabled
        assert face_system.enable_augmentation == True, "Augmentation should be enabled"
        assert face_system.augmentation_preset == "minimal", "Preset should be 'minimal'"
        print("✓ Augmentation settings verified")
        
        # Test that methods exist
        assert hasattr(face_system, 'enroll_with_augmentation'), "Missing enroll_with_augmentation method"
        assert hasattr(face_system, 'enroll_multiple_with_augmentation'), "Missing enroll_multiple_with_augmentation method"
        print("✓ Augmentation methods available")
        
        print("\n✓ Face system integration test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Face system integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augmentation_config():
    """Test augmentation configuration."""
    print("\n" + "=" * 60)
    print("TEST 3: Augmentation Configuration")
    print("=" * 60)
    
    # Test all presets exist
    presets = ["minimal", "balanced", "aggressive"]
    for preset in presets:
        config = AugmentationConfig.get_preset(preset)
        print(f"\n{preset.upper()} preset:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Verify required keys
        required_keys = ["num_lighting", "num_crops", "num_rotations", "add_blur", "add_noise"]
        for key in required_keys:
            assert key in config, f"Missing key '{key}' in {preset} preset"
    
    print("\n✓ Augmentation configuration test PASSED")
    return True


def test_augmentation_operations():
    """Test individual augmentation operations."""
    print("\n" + "=" * 60)
    print("TEST 4: Individual Augmentation Operations")
    print("=" * 60)
    
    augmentor = FaceAugmentation(save_augmented=False)
    test_image = create_test_image()
    
    # Test lighting variations
    print("\nTesting lighting variations...")
    lighting_imgs = augmentor._generate_lighting_variations(test_image, count=3)
    assert len(lighting_imgs) == 3, f"Expected 3 lighting images, got {len(lighting_imgs)}"
    print(f"✓ Generated {len(lighting_imgs)} lighting variations")
    
    # Test crop variations
    print("\nTesting crop variations...")
    crop_imgs = augmentor._generate_crop_variations(test_image, count=3)
    assert len(crop_imgs) == 3, f"Expected 3 crop images, got {len(crop_imgs)}"
    print(f"✓ Generated {len(crop_imgs)} crop variations")
    
    # Test rotation variations
    print("\nTesting rotation variations...")
    rotation_imgs = augmentor._generate_rotation_variations(test_image, count=3)
    assert len(rotation_imgs) == 3, f"Expected 3 rotation images, got {len(rotation_imgs)}"
    print(f"✓ Generated {len(rotation_imgs)} rotation variations")
    
    # Test blur variations
    print("\nTesting blur variations...")
    blur_imgs = augmentor._generate_blur_variations(test_image, count=2)
    assert len(blur_imgs) == 2, f"Expected 2 blur images, got {len(blur_imgs)}"
    print(f"✓ Generated {len(blur_imgs)} blur variations")
    
    # Test noise variations
    print("\nTesting noise variations...")
    noise_imgs = augmentor._generate_noise_variations(test_image, count=2)
    assert len(noise_imgs) == 2, f"Expected 2 noise images, got {len(noise_imgs)}"
    print(f"✓ Generated {len(noise_imgs)} noise variations")
    
    print("\n✓ Individual augmentation operations test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FACIAL DATA AUGMENTATION SYSTEM - TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run tests
    try:
        results.append(("Augmentation Module", test_augmentation_module()))
    except Exception as e:
        print(f"\n✗ Augmentation module test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Augmentation Module", False))
    
    try:
        results.append(("Face System Integration", test_face_system_integration()))
    except Exception as e:
        print(f"\n✗ Face system integration test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Face System Integration", False))
    
    try:
        results.append(("Augmentation Configuration", test_augmentation_config()))
    except Exception as e:
        print(f"\n✗ Augmentation configuration test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Augmentation Configuration", False))
    
    try:
        results.append(("Individual Operations", test_augmentation_operations()))
    except Exception as e:
        print(f"\n✗ Individual operations test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Individual Operations", False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests PASSED! Augmentation system is ready.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) FAILED. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
