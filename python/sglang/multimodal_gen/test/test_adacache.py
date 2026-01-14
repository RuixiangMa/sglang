# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.configs.sample.adacache import (
    AdaCacheParams,
    WanAdaCacheParams,
)
from sglang.multimodal_gen.runtime.utils.adacache import AdaCacheManager


class TestAdaCacheManager(unittest.TestCase):
    """Test cases for AdaCacheManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = AdaCacheParams(
            adacache_threshold=0.1,
            adacache_decay_factor=0.9,
            adacache_growth_factor=1.1,
            adacache_min_threshold=0.05,
            adacache_max_threshold=0.5,
            adacache_warmup_steps=2,
            adacache_diff_method="combined",
        )
        self.manager = AdaCacheManager(self.params)

    def test_initialization(self):
        """Test that AdaCacheManager initializes correctly."""
        self.assertEqual(self.manager.threshold, 0.1)
        self.assertEqual(self.manager.cache_hits, 0)
        self.assertEqual(self.manager.cache_misses, 0)
        self.assertIsNone(self.manager.previous_features)
        self.assertIsNone(self.manager.previous_residual)

    def test_compute_feature_difference_l2(self):
        """Test L2 feature difference computation."""
        self.params.adacache_diff_method = "l2"
        self.manager = AdaCacheManager(self.params)

        features1 = torch.randn(1, 64, 32, 32)
        features2 = features1 + 0.01 * torch.randn_like(features1)

        diff = self.manager.compute_feature_difference(features2, features1)
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 1.0)

    def test_compute_feature_difference_cosine(self):
        """Test cosine similarity feature difference computation."""
        self.params.adacache_diff_method = "cosine"
        self.manager = AdaCacheManager(self.params)

        features1 = torch.randn(1, 64, 32, 32)
        features2 = features1 + 0.01 * torch.randn_like(features1)

        diff = self.manager.compute_feature_difference(features2, features1)
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 1.0)

    def test_compute_feature_difference_combined(self):
        """Test combined feature difference computation."""
        features1 = torch.randn(1, 64, 32, 32)
        features2 = features1 + 0.01 * torch.randn_like(features1)

        diff = self.manager.compute_feature_difference(features2, features1)
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 1.0)

    def test_update_threshold_cache_hit(self):
        """Test threshold update on cache hit."""
        initial_threshold = self.manager.threshold
        self.manager.update_threshold(used_cache=True)
        
        expected_threshold = initial_threshold * self.params.adacache_decay_factor
        self.assertAlmostEqual(self.manager.threshold, expected_threshold, places=4)

    def test_update_threshold_cache_miss(self):
        """Test threshold update on cache miss."""
        initial_threshold = self.manager.threshold
        self.manager.update_threshold(used_cache=False)
        
        expected_threshold = initial_threshold * self.params.adacache_growth_factor
        self.assertAlmostEqual(self.manager.threshold, expected_threshold, places=4)

    def test_threshold_clamping_min(self):
        """Test threshold clamping to minimum value."""
        self.manager.threshold = 0.06
        self.manager.update_threshold(used_cache=True)
        
        self.assertGreaterEqual(self.manager.threshold, self.params.adacache_min_threshold)

    def test_threshold_clamping_max(self):
        """Test threshold clamping to maximum value."""
        self.manager.threshold = 0.49
        self.manager.update_threshold(used_cache=False)
        
        self.assertLessEqual(self.manager.threshold, self.params.adacache_max_threshold)

    def test_should_use_cache_warmup(self):
        """Test that cache is not used during warmup period."""
        features = torch.randn(1, 64, 32, 32)
        
        for i in range(self.params.adacache_warmup_steps):
            use_cache = self.manager.should_use_cache(features, i, 10)
            self.assertFalse(use_cache)

    def test_should_use_cache_after_warmup(self):
        """Test cache decision after warmup period."""
        features1 = torch.randn(1, 64, 32, 32)
        features2 = features1 + 0.001 * torch.randn_like(features1)
        
        # First, warmup steps
        for i in range(self.params.adacache_warmup_steps):
            use_cache = self.manager.should_use_cache(features1, i, 10)
            self.assertFalse(use_cache)
        
        # Cache features after warmup
        self.manager.cache_features(features1, features1)
        
        # Second step after warmup: should use cache (low difference)
        use_cache = self.manager.should_use_cache(features2, self.params.adacache_warmup_steps, 10)
        self.assertTrue(use_cache)

    def test_should_use_cache_high_difference(self):
        """Test that cache is not used with high feature difference."""
        features1 = torch.randn(1, 64, 32, 32)
        features2 = features1 + 0.5 * torch.randn_like(features1)
        
        # Cache features
        self.manager.cache_features(features1, features1)
        
        # High difference: should not use cache
        use_cache = self.manager.should_use_cache(features2, 3, 10)
        self.assertFalse(use_cache)

    def test_cache_features(self):
        """Test feature caching."""
        original_features = torch.randn(1, 64, 32, 32)
        computed_features = original_features + 0.01 * torch.randn_like(original_features)
        
        self.manager.cache_features(computed_features, original_features)
        
        self.assertIsNotNone(self.manager.previous_features)
        self.assertIsNotNone(self.manager.previous_residual)

    def test_retrieve_cached_states(self):
        """Test cached state retrieval."""
        original_features = torch.randn(1, 64, 32, 32)
        computed_features = original_features + 0.01 * torch.randn_like(original_features)
        
        self.manager.cache_features(computed_features, original_features)
        
        hidden_states = torch.randn(1, 64, 32, 32)
        retrieved = self.manager.retrieve_cached_states(hidden_states)
        
        expected = hidden_states + self.manager.previous_residual
        self.assertTrue(torch.allclose(retrieved, expected, atol=1e-5))

    def test_get_cache_stats(self):
        """Test cache statistics retrieval."""
        features1 = torch.randn(1, 64, 32, 32)
        features2 = features1 + 0.001 * torch.randn_like(features1)
        
        # First, warmup steps
        for i in range(self.params.adacache_warmup_steps):
            use_cache = self.manager.should_use_cache(features1, i, 10)
            self.assertFalse(use_cache)
        
        # Cache features after warmup
        self.manager.cache_features(features1, features1)
        
        # Hit (low difference)
        use_cache = self.manager.should_use_cache(features2, self.params.adacache_warmup_steps, 10)
        self.assertTrue(use_cache)
        
        # Miss (high difference)
        features3 = features1 + 0.5 * torch.randn_like(features1)
        use_cache = self.manager.should_use_cache(features3, self.params.adacache_warmup_steps + 1, 10)
        self.assertFalse(use_cache)
        
        stats = self.manager.get_cache_stats()
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5, places=2)

    def test_reset(self):
        """Test cache state reset."""
        features = torch.randn(1, 64, 32, 32)
        self.manager.cache_features(features, features)
        
        self.manager.reset()
        
        self.assertIsNone(self.manager.previous_features)
        self.assertIsNone(self.manager.previous_residual)
        self.assertEqual(self.manager.cache_hits, 0)
        self.assertEqual(self.manager.cache_misses, 0)
        self.assertEqual(self.manager.threshold, self.params.adacache_threshold)


class TestWanAdaCacheManager(unittest.TestCase):
    """Test cases for Wan-specific AdaCacheManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = WanAdaCacheParams(
            adacache_threshold=0.1,
            adacache_decay_factor=0.9,
            adacache_growth_factor=1.1,
            adacache_min_threshold=0.05,
            adacache_max_threshold=0.5,
            adacache_warmup_steps=2,
            adacache_diff_method="combined",
            use_ret_steps=True,
        )
        self.manager = AdaCacheManager(self.params)

    def test_wan_even_odd_alternation(self):
        """Test even/odd alternation for Wan models."""
        features = torch.randn(1, 64, 32, 32)
        
        # Step 0 (even)
        use_cache = self.manager.should_use_cache_wan(features, 0, 50)
        self.assertFalse(use_cache)
        self.assertTrue(self.manager.is_even)
        
        # Step 1 (odd)
        use_cache = self.manager.should_use_cache_wan(features, 1, 50)
        self.assertFalse(use_cache)
        self.assertFalse(self.manager.is_even)

    def test_wan_cache_features_even(self):
        """Test Wan feature caching for even steps."""
        features = torch.randn(1, 64, 32, 32)
        self.manager.is_even = True
        
        self.manager.cache_features_wan(features, features)
        
        self.assertIsNotNone(self.manager.previous_features_even)
        self.assertIsNotNone(self.manager.previous_residual_even)
        self.assertIsNone(self.manager.previous_features_odd)

    def test_wan_cache_features_odd(self):
        """Test Wan feature caching for odd steps."""
        features = torch.randn(1, 64, 32, 32)
        self.manager.is_even = False
        
        self.manager.cache_features_wan(features, features)
        
        self.assertIsNotNone(self.manager.previous_features_odd)
        self.assertIsNotNone(self.manager.previous_residual_odd)
        self.assertIsNone(self.manager.previous_features_even)

    def test_wan_retrieve_cached_states_even(self):
        """Test Wan cached state retrieval for even steps."""
        features = torch.randn(1, 64, 32, 32)
        residual = torch.randn(64, 32, 32)
        self.manager.is_even = True
        self.manager.previous_residual_even = residual
        
        hidden_states = torch.randn(1, 64, 32, 32)
        retrieved = self.manager.retrieve_cached_states_wan(hidden_states)
        
        expected = hidden_states + residual
        self.assertTrue(torch.allclose(retrieved, expected, atol=1e-5))

    def test_wan_retrieve_cached_states_odd(self):
        """Test Wan cached state retrieval for odd steps."""
        features = torch.randn(1, 64, 32, 32)
        residual = torch.randn(64, 32, 32)
        self.manager.is_even = False
        self.manager.previous_residual_odd = residual
        
        hidden_states = torch.randn(1, 64, 32, 32)
        retrieved = self.manager.retrieve_cached_states_wan(hidden_states)
        
        expected = hidden_states + residual
        self.assertTrue(torch.allclose(retrieved, expected, atol=1e-5))

    def test_wan_ret_steps(self):
        """Test Wan retention steps behavior."""
        features = torch.randn(1, 64, 32, 32)
        ret_steps = self.params.ret_steps
        cutoff_steps = self.params.get_cutoff_steps(50)
        
        # During ret_steps: always compute
        for i in range(ret_steps):
            use_cache = self.manager.should_use_cache_wan(features, i, 50)
            self.assertFalse(use_cache)
        
        # After ret_steps: can use cache
        self.manager.cache_features_wan(features, features)
        use_cache = self.manager.should_use_cache_wan(features, ret_steps, 50)
        self.assertTrue(use_cache)

    def test_wan_cutoff_steps(self):
        """Test Wan cutoff steps behavior."""
        features = torch.randn(1, 64, 32, 32)
        num_inference_steps = 50
        cutoff_steps = self.params.get_cutoff_steps(num_inference_steps)
        
        # At cutoff: always compute
        use_cache = self.manager.should_use_cache_wan(features, cutoff_steps - 1, num_inference_steps)
        self.assertFalse(use_cache)


class TestAdaCacheParams(unittest.TestCase):
    """Test cases for AdaCacheParams."""

    def test_default_values(self):
        """Test default parameter values."""
        params = AdaCacheParams()
        
        self.assertEqual(params.cache_type, "adacache")
        self.assertEqual(params.adacache_threshold, 0.1)
        self.assertEqual(params.adacache_decay_factor, 0.9)
        self.assertEqual(params.adacache_growth_factor, 1.1)
        self.assertEqual(params.adacache_min_threshold, 0.05)
        self.assertEqual(params.adacache_max_threshold, 0.5)
        self.assertEqual(params.adacache_warmup_steps, 2)
        self.assertEqual(params.adacache_diff_method, "combined")

    def test_custom_values(self):
        """Test custom parameter values."""
        params = AdaCacheParams(
            adacache_threshold=0.15,
            adacache_decay_factor=0.85,
            adacache_growth_factor=1.15,
            adacache_min_threshold=0.03,
            adacache_max_threshold=0.6,
            adacache_warmup_steps=3,
            adacache_diff_method="l2",
        )
        
        self.assertEqual(params.adacache_threshold, 0.15)
        self.assertEqual(params.adacache_decay_factor, 0.85)
        self.assertEqual(params.adacache_growth_factor, 1.15)
        self.assertEqual(params.adacache_min_threshold, 0.03)
        self.assertEqual(params.adacache_max_threshold, 0.6)
        self.assertEqual(params.adacache_warmup_steps, 3)
        self.assertEqual(params.adacache_diff_method, "l2")


class TestWanAdaCacheParams(unittest.TestCase):
    """Test cases for WanAdaCacheParams."""

    def test_default_values(self):
        """Test default Wan parameter values."""
        params = WanAdaCacheParams()
        
        self.assertEqual(params.cache_type, "adacache_wan")
        self.assertTrue(params.use_ret_steps)

    def test_ret_steps_property(self):
        """Test ret_steps property."""
        params = WanAdaCacheParams(use_ret_steps=True)
        self.assertEqual(params.ret_steps, 10)
        
        params = WanAdaCacheParams(use_ret_steps=False)
        self.assertEqual(params.ret_steps, 2)

    def test_get_cutoff_steps(self):
        """Test get_cutoff_steps method."""
        params = WanAdaCacheParams(use_ret_steps=True)
        cutoff = params.get_cutoff_steps(50)
        self.assertEqual(cutoff, 100)
        
        params = WanAdaCacheParams(use_ret_steps=False)
        cutoff = params.get_cutoff_steps(50)
        self.assertEqual(cutoff, 98)


if __name__ == "__main__":
    unittest.main()
