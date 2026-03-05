"""
Unit tests for Qwen-Image batch processing (batch_size > 1) support.
"""
import unittest
from PIL import Image


class TestNestedImageListDetection(unittest.TestCase):
    """Tests for _is_nested_image_list utility function."""
    
    def test_flat_list_returns_false(self):
        """Flat list [img1, img2] should return False (batch == 1)."""
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _is_nested_image_list
        
        img1 = Image.new("RGB", (64, 64))
        img2 = Image.new("RGB", (64, 64))
        
        self.assertFalse(_is_nested_image_list([img1, img2]))
    
    def test_nested_list_returns_true(self):
        """Nested list [[img1], [img2]] should return True (batch > 1)."""
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _is_nested_image_list
        
        img1 = Image.new("RGB", (64, 64))
        img2 = Image.new("RGB", (64, 64))
        
        self.assertTrue(_is_nested_image_list([[img1], [img2]]))
    
    def test_empty_list_returns_false(self):
        """Empty list should return False."""
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _is_nested_image_list
        
        self.assertFalse(_is_nested_image_list([]))
    
    def test_single_nested_list_returns_true(self):
        """Single-item nested list [[img1]] should return True (batch == 1 with nested format)."""
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _is_nested_image_list
        
        img1 = Image.new("RGB", (64, 64))
        
        self.assertTrue(_is_nested_image_list([[img1]]))
    
    def test_mixed_nested_list_returns_true(self):
        """Nested list with different lengths [[img1], [img2, img3]] should return True."""
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _is_nested_image_list
        
        img1 = Image.new("RGB", (64, 64))
        img2 = Image.new("RGB", (64, 64))
        img3 = Image.new("RGB", (64, 64))
        
        self.assertTrue(_is_nested_image_list([[img1], [img2, img3]]))


class TestPreprocessVaeImageLogic(unittest.TestCase):
    """Tests for preprocess_vae_image logic without full batch object."""
    
    def test_nested_detection_logic(self):
        """Verify nested detection logic matches expected behavior."""
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _is_nested_image_list
        
        # batch == 1 cases (flat lists)
        flat_cases = [
            [Image.new("RGB", (64, 64))],
            [Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))],
        ]
        for case in flat_cases:
            self.assertFalse(_is_nested_image_list(case), f"Failed for flat case: {len(case)} images")
        
        # batch > 1 cases (nested lists)
        nested_cases = [
            [[Image.new("RGB", (64, 64))], [Image.new("RGB", (64, 64))]],
            [[Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64))], [Image.new("RGB", (64, 64))]],
        ]
        for case in nested_cases:
            self.assertTrue(_is_nested_image_list(case), f"Failed for nested case: {len(case)} batch items")


if __name__ == "__main__":
    unittest.main()