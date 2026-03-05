# Qwen-Image Edit Batch Size > 1 Support Design

## Overview

Add batch processing support to SGLang's Qwen-Image Edit pipelines (`QwenImageEditPipelineConfig`, `QwenImageEditPlusPipelineConfig`, `QwenImageEditPlus_2511_PipelineConfig`) following the approach in [diffusers PR #12968](https://github.com/huggingface/diffusers/pull/12968).

## Current Limitation

```python
# qwen_image.py:435
def _prepare_edit_cond_kwargs(self, batch, prompt_embeds, rotary_emb, device, dtype):
    batch_size = batch.latents.shape[0]
    assert batch_size == 1  # ❌ Hard-coded limitation
    
    # All batch items share the same vae_image_sizes
    img_shapes = [
        [...for vae_width, vae_height in batch.vae_image_sizes],
    ] * batch_size  # ❌ Same shape repeated for all batch items
```

## Design Goals

1. Support `batch_size > 1` for Qwen-Image Edit pipelines
2. Each batch item can have different input images with different sizes
3. Maintain backward compatibility with existing `batch_size == 1` usage
4. Follow SGLang's coding patterns and conventions

## Data Flow Changes

### Input Format

```python
# batch_size == 1: single prompt with multiple reference images (FLAT list)
batch.condition_image = [img1, img2]  # Current format - unchanged

# batch_size > 1: multiple prompts, each with its own reference images (NESTED list)
batch.condition_image = [[img1_for_prompt1], [img1_for_prompt2, img2_for_prompt2]]
```

### vae_image_sizes Format

```python
# batch_size == 1 (flat)
batch.vae_image_sizes = [(w1, h1), (w2, h2)]  # Current format - unchanged

# batch_size > 1 (nested)
batch.vae_image_sizes = [[(w1, h1)], [(w1, h1), (w2, h2)]]  # New format
```

### img_shapes Construction

```python
# Current (problematic for batch > 1)
img_shapes = [
    [noisy_img_shape, *vae_img_shapes],
] * batch_size  # All batch items share same shape

# New (correct for batch > 1)
sizes_list = batch.vae_image_sizes if is_nested else [batch.vae_image_sizes]
img_shapes = [
    [noisy_img_shape, *[(1, h//f, w//f) for w, h in batch_vae_sizes]]
    for batch_vae_sizes in sizes_list
]
```

## Implementation Plan

### Phase 1: Core Pipeline Config Changes

**File:** `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`

#### 1.1 Add utility function for nested list detection

```python
def _is_nested_image_list(images: list) -> bool:
    """Check if images is a nested list (batch > 1) or flat list (batch == 1)."""
    return images and isinstance(images[0], list)
```

#### 1.2 Modify `preprocess_vae_image()` (line 418-429)

```python
def preprocess_vae_image(self, batch, vae_image_processor):
    if not isinstance(batch.condition_image, list):
        batch.condition_image = [batch.condition_image]
    
    is_nested = _is_nested_image_list(batch.condition_image)
    
    if is_nested:
        # batch_size > 1: each item has its own list of images
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

#### 1.3 Modify `_prepare_edit_cond_kwargs()` in `QwenImageEditPipelineConfig` (line 237-264)

```python
def _prepare_edit_cond_kwargs(self, batch, prompt_embeds, rotary_emb, device, dtype):
    batch_size = batch.latents.shape[0]
    # Remove: assert batch_size == 1
    
    height = batch.height
    width = batch.width
    image_size = batch.original_condition_image_size
    edit_width, edit_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
    vae_scale_factor = self.get_vae_scale_factor()

    # Handle both nested (batch > 1) and flat (batch == 1) formats
    is_nested = isinstance(batch.vae_image_sizes[0], list) if batch.vae_image_sizes else False
    sizes_list = batch.vae_image_sizes if is_nested else [batch.vae_image_sizes]
    
    img_shapes = [
        [
            (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2),
            (
                1,
                edit_height // vae_scale_factor // 2,
                edit_width // vae_scale_factor // 2,
            ),
        ]
        for _ in sizes_list
    ]
    txt_seq_lens = [prompt_embeds[0].shape[1]] * batch_size

    freqs_cis = QwenImageEditPipelineConfig.get_freqs_cis(
        img_shapes, txt_seq_lens, rotary_emb, device, dtype
    )
    # ... rest of the method
```

#### 1.4 Modify `_prepare_edit_cond_kwargs()` in `QwenImageEditPlusPipelineConfig` (line 431-493)

```python
def _prepare_edit_cond_kwargs(self, batch, prompt_embeds, rotary_emb, device, dtype):
    batch_size = batch.latents.shape[0]
    # Remove: assert batch_size == 1
    
    height = batch.height
    width = batch.width
    vae_scale_factor = self.get_vae_scale_factor()

    # Handle both nested (batch > 1) and flat (batch == 1) formats
    is_nested = isinstance(batch.vae_image_sizes[0], list) if batch.vae_image_sizes else False
    sizes_list = batch.vae_image_sizes if is_nested else [batch.vae_image_sizes]
    
    # Build img_shapes for each batch item separately
    img_shapes = [
        [
            (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2),
            *[
                (1, vae_height // vae_scale_factor // 2, vae_width // vae_scale_factor // 2)
                for vae_width, vae_height in batch_vae_sizes
            ],
        ]
        for batch_vae_sizes in sizes_list
    ]
    txt_seq_lens = [prompt_embeds[0].shape[1]] * batch_size

    freqs_cis = QwenImageEditPlusPipelineConfig.get_freqs_cis(
        img_shapes, txt_seq_lens, rotary_emb, device, dtype
    )
    # ... rest of the method
```

#### 1.5 Modify `_prepare_edit_cond_kwargs()` in `QwenImageEditPlus_2511_PipelineConfig` (line 506-540)

Similar changes as above.

### Phase 2: Input Validation Stage Changes

**File:** `python/sglang/multimodal_gen/runtime/pipelines_core/stages/input_validation.py`

The `preprocess_condition_image()` method (line 92-178) may need updates to handle nested lists for batch > 1.

### Phase 3: Denoising Stage Changes

**File:** `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`

Check if `batch.vae_image` and `batch.condition_image` usage in denoising stage supports nested lists.

### Phase 4: Testing

#### 4.1 Add unit tests for nested list handling

```python
# test_qwen_image_batch.py
def test_is_nested_image_list():
    assert _is_nested_image_list([[img1], [img2]]) == True
    assert _is_nested_image_list([img1, img2]) == False

def test_preprocess_vae_image_nested():
    # Test batch > 1 with nested list
    batch.condition_image = [[img1], [img2]]
    config.preprocess_vae_image(batch, vae_processor)
    assert isinstance(batch.vae_image_sizes[0], list)
    assert len(batch.vae_image_sizes) == 2

def test_img_shapes_batch_gt_1():
    # Verify img_shapes built correctly for each batch item
    pass
```

#### 4.2 Integration tests

```python
def test_qwen_image_edit_batch_2():
    generator = DiffGenerator.from_pretrained("Qwen/Qwen-Image-Edit-2509")
    
    images = generator.generate(
        sampling_params_kwargs=dict(
            prompt=["sunset version", "winter version"],
            image=[[mountain_img], [mountain_img]],  # Nested for batch=2
            num_inference_steps=50,
        )
    )
    assert len(images) == 2
```

## Files to Modify

| File | Changes |
|------|---------|
| `configs/pipeline_configs/qwen_image.py` | Remove `assert batch_size == 1`, add nested list handling |
| `runtime/pipelines_core/stages/input_validation.py` | Support nested `condition_image` lists |
| `runtime/pipelines_core/stages/denoising.py` | Handle nested `vae_image` if needed |

## Backward Compatibility

- `batch_size == 1` usage remains unchanged (flat list format)
- No breaking changes to existing API
- New nested list format only used when `batch_size > 1`

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Different image sizes per batch item | Ensure `img_shapes` built per-item, not shared |
| Memory usage for batch > 1 | Document recommended batch sizes |
| VAE encoding of nested images | Test thoroughly in Phase 3 |

## Success Criteria

1. ✅ `batch_size > 1` works without assertion error
2. ✅ Each batch item can have different number of reference images
3. ✅ Backward compatible with `batch_size == 1` usage
4. ✅ All existing tests pass
5. ✅ New batch tests pass