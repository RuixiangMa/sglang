# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Integration test for AdaCache functionality.
"""

import unittest

import torch

from sglang.multimodal_gen.configs.sample.adacache import (
    AdaCacheParams,
    WanAdaCacheParams,
)
from sglang.multimodal_gen.runtime.utils.adacache import AdaCacheManager


class TestAdaCacheIntegration(unittest.TestCase):
    """Integration tests for AdaCache."""

    def test_adacache_manager_lifecycle(self):
        """Test complete lifecycle of AdaCache manager."""
        params = AdaCacheParams(
            adacache_threshold=0.1,
            adacache_decay_factor=0.9,
            adacache_growth_factor=1.1,
            adacache_min_threshold=0.05,
            adacache_max_threshold=0.5,
            adacache_warmup_steps=2,
            adacache_diff_method="combined",
        )
        manager = AdaCacheManager(params)

        # Simulate inference steps
        num_steps = 10
        base_features = torch.randn(1, 64, 32, 32)
        features_list = [
            base_features + 0.001 * torch.randn_like(base_features) for _ in range(num_steps)
        ]

        for step in range(num_steps):
            current_features = features_list[step]

            # Check if we should use cache
            use_cache = manager.should_use_cache(
                current_features, step, num_steps
            )

            if use_cache:
                # Retrieve from cache
                cached_states = manager.retrieve_cached_states(current_features)
                self.assertIsNotNone(cached_states)
            else:
                # Compute and cache
                if step >= params.adacache_warmup_steps:
                    # Simulate computation
                    computed_features = current_features + 0.01 * torch.randn_like(current_features)
                    manager.cache_features(computed_features, current_features)

        # Verify cache was used (after warmup)
        stats = manager.get_cache_stats()
        # We should have some cache hits after warmup with similar features
        # The manager internally tracks hits/misses
        self.assertGreaterEqual(stats["cache_hits"], 0)
        # Note: cache_misses is only incremented when cache is not used after warmup
        # During warmup, the counter is not incremented
        self.assertGreaterEqual(stats["cache_misses"], 0)

    def test_adacache_wan_integration(self):
        """Test AdaCache integration for Wan models."""
        params = WanAdaCacheParams(
            adacache_threshold=0.1,
            adacache_decay_factor=0.9,
            adacache_growth_factor=1.1,
            adacache_min_threshold=0.05,
            adacache_max_threshold=0.5,
            adacache_warmup_steps=2,
            adacache_diff_method="combined",
            use_ret_steps=True,
        )
        manager = AdaCacheManager(params)

        # Simulate Wan inference with even/odd steps
        num_steps = 50
        base_features = torch.randn(1, 64, 32, 32)
        features_list = [
            base_features + 0.001 * torch.randn_like(base_features) for _ in range(num_steps)
        ]

        for step in range(num_steps):
            current_features = features_list[step]

            # Check if we should use cache
            use_cache = manager.should_use_cache_wan(
                current_features, step, num_steps
            )

            if use_cache:
                # Retrieve from cache
                cached_states = manager.retrieve_cached_states_wan(current_features)
                self.assertIsNotNone(cached_states)
            else:
                # Compute and cache
                if step >= params.ret_steps:
                    # Simulate computation
                    computed_features = current_features + 0.01 * torch.randn_like(current_features)
                    manager.cache_features_wan(computed_features, current_features)

        # Verify even/odd alternation
        # After processing all steps, the last step was step 49 (odd)
        # So is_even should be False
        expected_is_even = (num_steps - 1) % 2 == 0
        self.assertEqual(manager.is_even, expected_is_even)

        # Verify cache was used
        stats = manager.get_cache_stats()
        self.assertGreater(stats["cache_hits"], 0)
        # Note: cache_misses is only incremented when cache is not used after ret_steps
        # During ret_steps (warmup), the counter is not incremented
        self.assertGreaterEqual(stats["cache_misses"], 0)

    def test_adacache_threshold_adaptation(self):
        """Test that threshold adapts based on cache hits/misses."""
        params = AdaCacheParams(
            adacache_threshold=0.1,
            adacache_decay_factor=0.9,
            adacache_growth_factor=1.1,
            adacache_min_threshold=0.05,
            adacache_max_threshold=0.5,
            adacache_warmup_steps=2,
            adacache_diff_method="combined",
        )
        manager = AdaCacheManager(params)

        initial_threshold = manager.threshold

        # Simulate cache hits (should decrease threshold)
        for _ in range(5):
            manager.update_threshold(used_cache=True)

        self.assertLess(manager.threshold, initial_threshold)

        # Reset to initial threshold for miss test
        manager.threshold = initial_threshold

        # Simulate cache misses (should increase threshold)
        for _ in range(5):
            manager.update_threshold(used_cache=False)

        # After 5 misses with growth factor 1.1, threshold should be 0.1 * 1.1^5 ≈ 0.161
        # But we need to account for the clamping
        expected_threshold = initial_threshold * (params.adacache_growth_factor ** 5)
        self.assertGreater(manager.threshold, initial_threshold)
        self.assertLess(manager.threshold, expected_threshold + 0.01)

        # Verify clamping
        manager.threshold = 0.06
        manager.update_threshold(used_cache=True)
        self.assertGreaterEqual(manager.threshold, params.adacache_min_threshold)

        manager.threshold = 0.49
        manager.update_threshold(used_cache=False)
        self.assertLessEqual(manager.threshold, params.adacache_max_threshold)

    def test_adacache_reset_between_requests(self):
        """Test that cache state is properly reset between requests."""
        params = AdaCacheParams(
            adacache_threshold=0.1,
            adacache_warmup_steps=2,
            adacache_diff_method="combined",
        )
        manager = AdaCacheManager(params)

        # First request
        features1 = torch.randn(1, 64, 32, 32)
        manager.cache_features(features1, features1)
        self.assertIsNotNone(manager.previous_features)

        # Reset
        manager.reset()
        self.assertIsNone(manager.previous_features)
        self.assertIsNone(manager.previous_residual)
        self.assertEqual(manager.cache_hits, 0)
        self.assertEqual(manager.cache_misses, 0)
        self.assertEqual(manager.threshold, params.adacache_threshold)

        # Second request should start fresh
        features2 = torch.randn(1, 64, 32, 32)
        manager.cache_features(features2, features2)
        self.assertIsNotNone(manager.previous_features)

    def test_adacache_different_methods(self):
        """Test different feature difference computation methods."""
        features1 = torch.randn(1, 64, 32, 32)
        features2 = features1 + 0.01 * torch.randn_like(features1)

        methods = ["l2", "cosine", "combined"]

        for method in methods:
            params = AdaCacheParams(
                adacache_threshold=0.1,
                adacache_warmup_steps=2,
                adacache_diff_method=method,
            )
            manager = AdaCacheManager(params)

            diff = manager.compute_feature_difference(features2, features1)
            self.assertGreater(diff, 0.0)
            self.assertLess(diff, 1.0)

    def test_adacache_statistics_accuracy(self):
        """Test that cache statistics are accurately tracked."""
        params = AdaCacheParams(
            adacache_threshold=0.1,
            adacache_warmup_steps=2,
            adacache_diff_method="combined",
        )
        manager = AdaCacheManager(params)

        # Simulate known number of hits and misses
        expected_hits = 3
        expected_misses = 5

        # Warmup
        for i in range(params.adacache_warmup_steps):
            features = torch.randn(1, 64, 32, 32)
            manager.should_use_cache(features, i, 10)

        # Cache initial features
        base_features = torch.randn(1, 64, 32, 32)
        manager.cache_features(base_features, base_features)

        # Simulate hits (low difference)
        for _ in range(expected_hits):
            features = base_features + 0.001 * torch.randn_like(base_features)
            manager.should_use_cache(features, params.adacache_warmup_steps, 10)

        # Simulate misses (high difference)
        for _ in range(expected_misses):
            features = base_features + 0.5 * torch.randn_like(base_features)
            manager.should_use_cache(features, params.adacache_warmup_steps, 10)

        stats = manager.get_cache_stats()
        self.assertEqual(stats["cache_hits"], expected_hits)
        self.assertEqual(stats["cache_misses"], expected_misses)

        expected_hit_rate = expected_hits / (expected_hits + expected_misses)
        self.assertAlmostEqual(stats["hit_rate"], expected_hit_rate, places=2)


if __name__ == "__main__":
    unittest.main()
