# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


@dataclass
class AdaCacheParams(CacheParams):
    """
    AdaCache (Adaptive Cache) parameters for runtime-adaptive caching.
    
    AdaCache provides 2.6-4.7x speedup with minimal quality loss by
    dynamically adjusting caching strategy based on feature differences.
    """
    cache_type: str = "adacache"
    
    # Initial threshold for feature difference (0.05 - 0.5)
    # Lower values = more aggressive caching, higher values = more conservative
    adacache_threshold: float = 0.1
    
    # Decay factor for threshold when cache is hit (0.8 - 0.95)
    # Makes caching more aggressive after successful cache hits
    adacache_decay_factor: float = 0.9
    
    # Growth factor for threshold when cache is missed (1.05 - 1.15)
    # Makes caching more conservative after cache misses
    adacache_growth_factor: float = 1.1
    
    # Minimum threshold to prevent overly aggressive caching
    adacache_min_threshold: float = 0.05
    
    # Maximum threshold to prevent overly conservative caching
    adacache_max_threshold: float = 0.5
    
    # Number of warmup steps before caching starts
    adacache_warmup_steps: int = 2
    
    # Feature difference computation method: "l2", "cosine", "combined"
    adacache_diff_method: str = "combined"
    
    # Enable/disable AdaCache
    enable_adacache: bool = False


@dataclass
class WanAdaCacheParams(AdaCacheParams):
    """
    Wan-specific AdaCache parameters.
    Wan models use even/odd alternating caching strategy.
    """
    cache_type: str = "adacache_wan"
    
    # Use retention steps (ret_steps) for Wan models
    use_ret_steps: bool = True
    
    @property
    def ret_steps(self) -> int:
        if self.use_ret_steps:
            return 5 * 2
        else:
            return 1 * 2
    
    def get_cutoff_steps(self, num_inference_steps: int) -> int:
        if self.use_ret_steps:
            return num_inference_steps * 2
        else:
            return num_inference_steps * 2 - 2
