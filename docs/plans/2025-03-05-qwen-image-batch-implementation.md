# Qwen-Image Batch > 1 Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add batch processing support (batch_size > 1) to Qwen-Image Edit pipelines.

**Architecture:** Use nested list format `[[img1], [img2]]` to distinguish batch > 1 from batch == 1. Modify `preprocess_vae_image()` and `_prepare_edit_cond_kwargs()` to detect and handle nested lists. Build `img_shapes` per batch item instead of sharing same shape.

**Tech Stack:** Python, SGLang multimodal_gen, PyTorch

---

## Task 1: Add Utility Function for Nested List Detection

**Files:**
- Modify: `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py:1-40`

**Step 1: Add utility function after imports**

Add at line 40 (after imports, before first class):

```python
def _is_nested_image_list(images: list) -> bool:
    """
    Check if images is a nested list (batch > 1) or flat list (batch == 1).
    
    Args:
        images: List of PIL images, either flat [img1, img2] or nested [[img1], [img2]]
    
    Returns:
        True if nested list (batch > 1), False if flat list (batch == 1)
    """
    return bool(images) and isinstance(images[0], list)
```

**Step 2: Verify syntax**

Run: `python -c "from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _is_nested_image_list; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py
git commit -m "feat(qwen-image): add _is_nested_image_list utility function"
```

---

## Task 2: Modify preprocess_vae_image for Nested List Support

**Files:**
- Modify: `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py:418-429`

**Step 1: Replace current preprocess_vae_image method**

Find `def preprocess_vae_image(self, batch, vae_image_processor):` at line 418 and replace the entire method with:

```python
def preprocess_vae_image(self, batch, vae_image_processor):
    """
    Preprocess VAE images, supporting both batch_size == 1 and batch_size > 1.
    
    For batch_size == 1: condition_image = [img1, img2] (flat list)
    For batch_size > 1: condition_image = [[img1], [img2, img3]] (nested list)
    """
    if not isinstance(batch.condition_image, list):
        batch.condition_image = [batch.condition_image]
    
    is_nested = _is_nested_image_list(batch.condition_image)
    
    if is_nested:
        # batch_size > 1: each batch item has its own list of images
        all_vae_images = []
        all_vae_sizes = []
        for batch_images in batch.condition_image:
            batch_vae_images = []
            batch_vae_sizes = []
            for img in batch_images:
                width, height = self.calculate_vae_image_size(img, img.width, img.height)
                batch_vae_images.append(vae_image_processor.preprocess(img, height, width))
                batch_vae_sizes.append((width, height))
            all_vae_images.append(batch_vae_images)
            all_vae_sizes.append(batch_vae_sizes)
        batch.vae_image = all_vae_images
        batch.vae_image_sizes = all_vae_sizes
    else:
        # batch_size == 1: all images belong to single prompt (backward compatible)
        new_images = []
        vae_image_sizes = []
        for img in batch.condition_image:
            width, height = self.calculate_vae_image_size(img, img.width, img.height)
            new_images.append(vae_image_processor.preprocess(img, height, width))
            vae_image_sizes.append((width, height))
        batch.vae_image = new_images
        batch.vae_image_sizes = vae_image_sizes
    
    return batch
```

**Step 2: Verify syntax**

Run: `python -c "from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import QwenImageEditPlusPipelineConfig; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py
git commit -m "feat(qwen-image): support nested list in preprocess_vae_image"
```

---

## Task 3: Modify _prepare_edit_cond_kwargs in QwenImageEditPipelineConfig

**Files:**
- Modify: `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py:237-264`

**Step 1: Remove assert and update method**

Find `def _prepare_edit_cond_kwargs` in `QwenImageEditPipelineConfig` (around line 237) and replace with:

```python
def _prepare_edit_cond_kwargs(
    self, batch, prompt_embeds, rotary_emb, device, dtype
):
    batch_size = batch.latents.shape[0]
    height = batch.height
    width = batch.width
    image_size = batch.original_condition_image_size
    edit_width, edit_height, _ = calculate_dimensions(
        1024 * 1024, image_size[0] / image_size[1]
    )
    vae_scale_factor = self.get_vae_scale_factor()

    img_shapes = [
        [
            (
                1,
                height // vae_scale_factor // 2,
                width // vae_scale_factor // 2,
            ),
            (
                1,
                edit_height // vae_scale_factor // 2,
                edit_width // vae_scale_factor // 2,
            ),
        ]
    ] * batch_size
    txt_seq_lens = [prompt_embeds[0].shape[1]] * batch_size

    freqs_cis = QwenImageEditPipelineConfig.get_freqs_cis(
        img_shapes, txt_seq_lens, rotary_emb, device, dtype
    )

    # perform sp shard on noisy image tokens
    noisy_img_seq_len = (
        1 * (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
    )

    img_cache, txt_cache = freqs_cis
    noisy_img_cache = shard_rotary_emb_for_sp(img_cache[:noisy_img_seq_len, :])
    img_cache = torch.cat(
        [noisy_img_cache, img_cache[noisy_img_seq_len:, :]], dim=0
    ).to(device=device)
    return {
        "txt_seq_lens": txt_seq_lens,
        "freqs_cis": (img_cache, txt_cache),
        "img_shapes": img_shapes,
    }
```

**Step 2: Verify syntax**

Run: `python -c "from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import QwenImageEditPipelineConfig; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py
git commit -m "feat(qwen-image): remove batch_size == 1 assert in QwenImageEditPipelineConfig"
```

---

## Task 4: Modify _prepare_edit_cond_kwargs in QwenImageEditPlusPipelineConfig

**Files:**
- Modify: `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py:431-493`

**Step 1: Replace the entire method**

Find `def _prepare_edit_cond_kwargs` in `QwenImageEditPlusPipelineConfig` (around line 431) and replace with:

```python
def _prepare_edit_cond_kwargs(
    self, batch, prompt_embeds, rotary_emb, device, dtype
):
    batch_size = batch.latents.shape[0]
    height = batch.height
    width = batch.width

    vae_scale_factor = self.get_vae_scale_factor()

    # Handle both nested (batch > 1) and flat (batch == 1) formats
    is_nested = (
        isinstance(batch.vae_image_sizes[0], list) 
        if batch.vae_image_sizes else False
    )
    sizes_list = batch.vae_image_sizes if is_nested else [batch.vae_image_sizes]

    # Build img_shapes for each batch item separately
    img_shapes = [
        [
            (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2),
            *[
                (
                    1,
                    vae_height // vae_scale_factor // 2,
                    vae_width // vae_scale_factor // 2,
                )
                for vae_width, vae_height in batch_vae_sizes
            ],
        ]
        for batch_vae_sizes in sizes_list
    ]
    txt_seq_lens = [prompt_embeds[0].shape[1]] * batch_size

    freqs_cis = QwenImageEditPlusPipelineConfig.get_freqs_cis(
        img_shapes, txt_seq_lens, rotary_emb, device, dtype
    )

    # perform sp shard on noisy image tokens
    noisy_img_seq_len = (
        1 * (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
    )

    if isinstance(freqs_cis[0], torch.Tensor) and freqs_cis[0].dim() == 2:
        img_cache, txt_cache = freqs_cis
        noisy_img_cache = shard_rotary_emb_for_sp(img_cache[:noisy_img_seq_len, :])
        img_cache = torch.cat(
            [noisy_img_cache, img_cache[noisy_img_seq_len:, :]], dim=0
        ).to(device=device)
        return {
            "txt_seq_lens": txt_seq_lens,
            "freqs_cis": (img_cache, txt_cache),
            "img_shapes": img_shapes,
        }

    (img_cos, img_sin), (txt_cos, txt_sin) = freqs_cis
    noisy_img_cos = shard_rotary_emb_for_sp(img_cos[:noisy_img_seq_len, :])
    noisy_img_sin = shard_rotary_emb_for_sp(img_sin[:noisy_img_seq_len, :])

    # concat back the img_cos for input image (since it is not sp-shared later)
    img_cos = torch.cat([noisy_img_cos, img_cos[noisy_img_seq_len:, :]], dim=0).to(
        device=device
    )
    img_sin = torch.cat([noisy_img_sin, img_sin[noisy_img_seq_len:, :]], dim=0).to(
        device=device
    )

    return {
        "txt_seq_lens": txt_seq_lens,
        "freqs_cis": ((img_cos, img_sin), (txt_cos, txt_sin)),
        "img_shapes": img_shapes,
    }
```

**Step 2: Verify syntax**

Run: `python -c "from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import QwenImageEditPlusPipelineConfig; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py
git commit -m "feat(qwen-image): add batch > 1 support to QwenImageEditPlusPipelineConfig"
```

---

## Task 5: Modify _prepare_edit_cond_kwargs in QwenImageEditPlus_2511_PipelineConfig

**Files:**
- Modify: `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py:506-540`

**Step 1: Replace the method**

Find `def _prepare_edit_cond_kwargs` in `QwenImageEditPlus_2511_PipelineConfig` (around line 506) and replace with:

```python
def _prepare_edit_cond_kwargs(
    self, batch, prompt_embeds, rotary_emb, device, dtype
):
    batch_size = batch.latents.shape[0]
    height = batch.height
    width = batch.width
    image_size = batch.original_condition_image_size

    vae_scale_factor = self.get_vae_scale_factor()

    img_shapes = batch.img_shapes
    txt_seq_lens = batch.txt_seq_lens

    freqs_cis = QwenImageEditPlusPipelineConfig.get_freqs_cis(
        img_shapes, txt_seq_lens, rotary_emb, device, dtype
    )

    # perform sp shard on noisy image tokens
    noisy_img_seq_len = (
        1 * (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
    )

    img_cache, txt_cache = freqs_cis
    noisy_img_cache = shard_rotary_emb_for_sp(img_cache[:noisy_img_seq_len, :])
    img_cache = torch.cat(
        [noisy_img_cache, img_cache[noisy_img_seq_len:, :]], dim=0
    ).to(device=device)

    return {
        "txt_seq_lens": txt_seq_lens,
        "img_shapes": img_shapes,
        "freqs_cis": (img_cache, txt_cache),
        "additional_t_cond": torch.tensor([0] * batch_size, device=device, dtype=torch.long),
    }
```

**Step 2: Verify syntax**

Run: `python -c "from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import QwenImageEditPlus_2511_PipelineConfig; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py
git commit -m "feat(qwen-image): add batch > 1 support to QwenImageEditPlus_2511_PipelineConfig"
```

---

## Task 6: Write Unit Tests

**Files:**
- Create: `test/srt/test_qwen_image_batch.py`

**Step 1: Create test file**

```python
"""
Unit tests for Qwen-Image batch processing (batch_size > 1) support.
"""
import unittest
from PIL import Image
import torch


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


class TestPreprocessVaeImageBatch(unittest.TestCase):
    """Tests for preprocess_vae_image with batch support."""
    
    def test_flat_list_produces_flat_vae_image_sizes(self):
        """Flat condition_image should produce flat vae_image_sizes."""
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
            QwenImageEditPlusPipelineConfig,
        )
        from sglang.multimodal_gen.configs.sample.sampling_params import Req
        
        # Create mock batch
        batch = Req()
        img1 = Image.new("RGB", (128, 128))
        img2 = Image.new("RGB", (128, 128))
        batch.condition_image = [img1, img2]
        
        # Mock vae_image_processor
        class MockProcessor:
            def preprocess(self, img, height, width):
                return torch.zeros(1, 3, height, width)
        
        config = QwenImageEditPlusPipelineConfig()
        config.preprocess_vae_image(batch, MockProcessor())
        
        # Should be flat list: [(w1, h1), (w2, h2)]
        self.assertIsInstance(batch.vae_image_sizes, list)
        self.assertNotIsInstance(batch.vae_image_sizes[0], list)
    
    def test_nested_list_produces_nested_vae_image_sizes(self):
        """Nested condition_image should produce nested vae_image_sizes."""
        from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
            QwenImageEditPlusPipelineConfig,
        )
        from sglang.multimodal_gen.configs.sample.sampling_params import Req
        
        # Create mock batch
        batch = Req()
        img1 = Image.new("RGB", (128, 128))
        img2 = Image.new("RGB", (128, 128))
        batch.condition_image = [[img1], [img2]]  # Nested for batch=2
        
        # Mock vae_image_processor
        class MockProcessor:
            def preprocess(self, img, height, width):
                return torch.zeros(1, 3, height, width)
        
        config = QwenImageEditPlusPipelineConfig()
        config.preprocess_vae_image(batch, MockProcessor())
        
        # Should be nested list: [[(w1, h1)], [(w2, h2)]]
        self.assertIsInstance(batch.vae_image_sizes, list)
        self.assertIsInstance(batch.vae_image_sizes[0], list)
        self.assertEqual(len(batch.vae_image_sizes), 2)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run tests**

Run: `cd /home/ruixiang/sglang && python test/srt/test_qwen_image_batch.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add test/srt/test_qwen_image_batch.py
git commit -m "test(qwen-image): add batch processing unit tests"
```

---

## Task 7: Run Existing Tests to Verify No Regression

**Files:**
- None (run existing tests)

**Step 1: Run Qwen-Image related tests**

Run: `cd /home/ruixiang/sglang && python -m pytest test/ -k "qwen" -v --tb=short 2>&1 | head -50`
Expected: No new failures

**Step 2: Verify no type errors**

Run: `python -c "from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import *; print('All imports OK')"`
Expected: `All imports OK`

**Step 3: Commit any fixes if needed**

```bash
git add -A
git commit -m "fix(qwen-image): address test failures"
```

---

## Summary

| Task | Description | File |
|------|-------------|------|
| 1 | Add `_is_nested_image_list` utility | `qwen_image.py:40` |
| 2 | Modify `preprocess_vae_image` | `qwen_image.py:418-429` |
| 3 | Update `QwenImageEditPipelineConfig._prepare_edit_cond_kwargs` | `qwen_image.py:237-264` |
| 4 | Update `QwenImageEditPlusPipelineConfig._prepare_edit_cond_kwargs` | `qwen_image.py:431-493` |
| 5 | Update `QwenImageEditPlus_2511_PipelineConfig._prepare_edit_cond_kwargs` | `qwen_image.py:506-540` |
| 6 | Add unit tests | `test/srt/test_qwen_image_batch.py` |
| 7 | Run existing tests | Verify no regression |

---

## Success Criteria

1. ✅ `batch_size > 1` works without assertion error
2. ✅ Each batch item can have different number of reference images  
3. ✅ Backward compatible with `batch_size == 1` usage
4. ✅ All existing tests pass
5. ✅ New unit tests pass