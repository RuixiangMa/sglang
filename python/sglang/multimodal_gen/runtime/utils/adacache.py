# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class AdaCacheManager:
    """
    AdaCache (Adaptive Cache) Manager
    
    Implements runtime-adaptive caching for DiT models with dynamic threshold adjustment.
    Provides 2.6-4.7x speedup with minimal quality loss.
    
    Key features:
    - Adaptive threshold: Adjusts cache threshold based on feature differences
    - Feature difference computation: Multiple methods (L2, cosine, combined)
    - Runtime decision: Dynamically decides whether to use cache or recompute
    """
    
    def __init__(self, adacache_params):
        self.params = adacache_params
        
        # Cache state
        self.cnt = 0
        self.threshold = self.params.adacache_threshold
        
        # Feature cache
        self.previous_features: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None
        self.accumulated_diff = 0.0
        
        # Wan-specific state (similar to TeaCache)
        self.is_even = False
        self.previous_e0_even: torch.Tensor | None = None
        self.previous_e0_odd: torch.Tensor | None = None
        self.previous_residual_even: torch.Tensor | None = None
        self.previous_residual_odd: torch.Tensor | None = None
        self.accumulated_diff_even = 0.0
        self.accumulated_diff_odd = 0.0
        self.should_calc_even = True
        self.should_calc_odd = True
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(
            f"AdaCache initialized with threshold={self.threshold:.4f}, "
            f"warmup_steps={self.params.adacache_warmup_steps}"
        )
    
    def compute_feature_difference(
        self, 
        current_features: torch.Tensor, 
        cached_features: torch.Tensor
    ) -> float:
        """
        Compute feature difference between current and cached features.
        
        Args:
            current_features: Current step features [B, C, H, W] or [B, L, C]
            cached_features: Cached features [B, C, H, W] or [B, L, C]
        
        Returns:
            diff_score: Normalized difference score [0, 1]
        """
        method = self.params.adacache_diff_method
        
        if method == "l2":
            # L2 distance normalized by mean absolute value
            l2_diff = torch.norm(current_features - cached_features, p=2)
            l2_norm = torch.norm(cached_features, p=2)
            diff_score = (l2_diff / (l2_norm + 1e-8)).item()
        
        elif method == "cosine":
            # Cosine similarity difference
            flat_current = current_features.flatten(1)
            flat_cached = cached_features.flatten(1)
            cosine_sim = F.cosine_similarity(flat_current, flat_cached, dim=-1)
            diff_score = (1.0 - torch.mean(cosine_sim)).item()
        
        elif method == "combined":
            # Combined L2 and cosine similarity
            l2_diff = torch.norm(current_features - cached_features, p=2)
            l2_norm = torch.norm(cached_features, p=2)
            l2_score = (l2_diff / (l2_norm + 1e-8)).item()
            
            flat_current = current_features.flatten(1)
            flat_cached = cached_features.flatten(1)
            cosine_sim = F.cosine_similarity(flat_current, flat_cached, dim=-1)
            cosine_score = (1.0 - torch.mean(cosine_sim)).item()
            
            # Weighted combination
            diff_score = 0.5 * l2_score + 0.5 * cosine_score
        
        else:
            raise ValueError(f"Unknown diff_method: {method}")
        
        return diff_score
    
    def update_threshold(self, used_cache: bool) -> None:
        """
        Update adaptive threshold based on cache hit/miss.
        
        Args:
            used_cache: Whether cache was used in this step
        """
        if used_cache:
            # Cache hit: make threshold more aggressive (lower)
            self.threshold *= self.params.adacache_decay_factor
        else:
            # Cache miss: make threshold more conservative (higher)
            self.threshold *= self.params.adacache_growth_factor
        
        # Clamp threshold to valid range
        self.threshold = max(
            self.params.adacache_min_threshold,
            min(self.params.adacache_max_threshold, self.threshold)
        )
    
    def should_use_cache(
        self, 
        current_features: torch.Tensor,
        current_timestep: int,
        num_inference_steps: int
    ) -> bool:
        """
        Decide whether to use cached features.
        
        Args:
            current_features: Current step features
            current_timestep: Current timestep index
            num_inference_steps: Total number of inference steps
        
        Returns:
            use_cache: Whether to use cached features
        """
        # Reset counter at start
        if current_timestep == 0:
            self.cnt = 0
            self.accumulated_diff = 0.0
            self.accumulated_diff_even = 0.0
            self.accumulated_diff_odd = 0.0
        
        # Warmup period: always compute
        if self.cnt < self.params.adacache_warmup_steps:
            self.cnt += 1
            return False
        
        # No cached features available
        if self.previous_features is None:
            self.cnt += 1
            return False
        
        # Compute feature difference
        diff_score = self.compute_feature_difference(
            current_features, self.previous_features
        )
        
        # Accumulate difference
        self.accumulated_diff += diff_score
        
        # Decision based on threshold
        use_cache = self.accumulated_diff < self.threshold
        
        if use_cache:
            self.cache_hits += 1
            logger.debug(
                f"AdaCache HIT at step {self.cnt}: "
                f"diff={diff_score:.4f}, accumulated={self.accumulated_diff:.4f}, "
                f"threshold={self.threshold:.4f}"
            )
        else:
            self.cache_misses += 1
            logger.debug(
                f"AdaCache MISS at step {self.cnt}: "
                f"diff={diff_score:.4f}, accumulated={self.accumulated_diff:.4f}, "
                f"threshold={self.threshold:.4f}"
            )
        
        self.cnt += 1
        return use_cache
    
    def should_use_cache_wan(
        self,
        current_features: torch.Tensor,
        current_timestep: int,
        num_inference_steps: int
    ) -> bool:
        """
        Wan-specific cache decision with even/odd alternating strategy.
        
        Args:
            current_features: Current step features
            current_timestep: Current timestep index
            num_inference_steps: Total number of inference steps
        
        Returns:
            use_cache: Whether to use cached features
        """
        # Reset counter at start
        if current_timestep == 0:
            self.cnt = 0
            self.accumulated_diff_even = 0.0
            self.accumulated_diff_odd = 0.0
            logger.info(f"AdaCache: Reset at timestep {current_timestep}")
        
        # Get Wan-specific parameters
        cutoff_steps = self.params.get_cutoff_steps(num_inference_steps)
        ret_steps = self.params.ret_steps
        
        logger.info(f"AdaCache: cnt={self.cnt}, current_timestep={current_timestep}, ret_steps={ret_steps}, cutoff_steps={cutoff_steps}")
        
        # Determine even/odd
        if self.cnt % 2 == 0:
            self.is_even = True
            
            # Always compute during warmup or after cutoff
            if self.cnt < ret_steps or self.cnt >= cutoff_steps:
                self.should_calc_even = True
                self.accumulated_diff_even = 0.0
                self.cnt += 1
                logger.info(f"AdaCache: Even step {self.cnt}, warmup or cutoff, no cache")
                return False
            
            # Check cache availability
            if self.previous_e0_even is None:
                self.should_calc_even = True
                self.accumulated_diff_even = 0.0
                self.previous_e0_even = current_features.clone()
                self.cnt += 1
                logger.info(f"AdaCache: Even step {self.cnt}, no cache available")
                return False
            
            # Compute feature difference
            diff_score = self.compute_feature_difference(
                current_features, self.previous_e0_even
            )
            
            # Accumulate difference
            self.accumulated_diff_even += diff_score
            
            # Decision based on threshold
            use_cache = self.accumulated_diff_even < self.params.adacache_threshold
            
            if use_cache:
                self.should_calc_even = False
                self.cache_hits += 1
                logger.info(
                    f"AdaCache HIT (even) at step {self.cnt}: "
                    f"diff={diff_score:.4f}, accumulated={self.accumulated_diff_even:.4f}, "
                    f"threshold={self.params.adacache_threshold:.4f}"
                )
            else:
                self.should_calc_even = True
                self.accumulated_diff_even = 0.0
                self.cache_misses += 1
                logger.info(
                    f"AdaCache MISS (even) at step {self.cnt}: "
                    f"diff={diff_score:.4f}, accumulated={self.accumulated_diff_even:.4f}, "
                    f"threshold={self.params.adacache_threshold:.4f}"
                )
            
            self.previous_e0_even = current_features.clone()
            self.cnt += 1
            return use_cache
        
        else:
            self.is_even = False
            
            # Always compute during warmup or after cutoff
            if self.cnt < ret_steps or self.cnt >= cutoff_steps:
                self.should_calc_odd = True
                self.accumulated_diff_odd = 0.0
                self.previous_e0_odd = current_features.clone()
                self.cnt += 1
                return False
            
            # Check cache availability
            if self.previous_e0_odd is None:
                self.should_calc_odd = True
                self.accumulated_diff_odd = 0.0
                self.previous_e0_odd = current_features.clone()
                self.cnt += 1
                return False
            
            # Compute feature difference
            diff_score = self.compute_feature_difference(
                current_features, self.previous_e0_odd
            )
            
            # Accumulate difference
            self.accumulated_diff_odd += diff_score
            
            # Decision based on threshold
            use_cache = self.accumulated_diff_odd < self.params.adacache_threshold
            
            if use_cache:
                self.should_calc_odd = False
                self.cache_hits += 1
                logger.debug(
                    f"AdaCache HIT (odd) at step {self.cnt}: "
                    f"diff={diff_score:.4f}, accumulated={self.accumulated_diff_odd:.4f}, "
                    f"threshold={self.params.adacache_threshold:.4f}"
                )
            else:
                self.should_calc_odd = True
                self.accumulated_diff_odd = 0.0
                self.cache_misses += 1
                logger.debug(
                    f"AdaCache MISS (odd) at step {self.cnt}: "
                    f"diff={diff_score:.4f}, accumulated={self.accumulated_diff_odd:.4f}, "
                    f"threshold={self.params.adacache_threshold:.4f}"
                )
            
            self.previous_e0_odd = current_features.clone()
            self.cnt += 1
            return use_cache
    
    def cache_features(
        self,
        features: torch.Tensor,
        original_features: torch.Tensor
    ) -> None:
        """
        Cache features for future use.
        
        Args:
            features: Computed features after forward pass
            original_features: Original features before forward pass
        """
        # Store residual (difference)
        self.previous_residual = features - original_features
        self.previous_features = original_features.clone()
        
        # Increment counter
        self.cnt += 1
    
    def cache_features_wan(
        self,
        features: torch.Tensor,
        original_features: torch.Tensor
    ) -> None:
        """
        Wan-specific feature caching with even/odd strategy.
        This caches the residual between features and original_features,
        similar to TeaCache behavior.
        
        Args:
            features: Computed features after forward pass
            original_features: Original features before forward pass
        """
        residual = features - original_features
        
        if self.is_even:
            self.previous_residual_even = residual.squeeze(0)
        else:
            self.previous_residual_odd = residual.squeeze(0)
        
        # Increment counter
        self.cnt += 1
    
    def retrieve_cached_states(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve cached states and apply residual.
        
        Args:
            hidden_states: Current hidden states
        
        Returns:
            updated_hidden_states: Hidden states with cached residual applied
        """
        if self.previous_residual is None:
            return hidden_states
        
        return hidden_states + self.previous_residual
    
    def retrieve_cached_states_wan(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Wan-specific cache retrieval with even/odd strategy.
        
        Args:
            hidden_states: Current hidden states
        
        Returns:
            updated_hidden_states: Hidden states with cached residual applied
        """
        if self.is_even:
            if self.previous_residual_even is None:
                return hidden_states
            return hidden_states + self.previous_residual_even
        else:
            if self.previous_residual_odd is None:
                return hidden_states
            return hidden_states + self.previous_residual_odd
    
    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            stats: Dictionary containing cache statistics
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "current_threshold": self.threshold,
            "total_steps": total
        }
    
    def reset(self) -> None:
        """Reset cache state."""
        self.cnt = 0
        self.threshold = self.params.adacache_threshold
        self.previous_features = None
        self.previous_residual = None
        self.accumulated_diff = 0.0
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.accumulated_diff_even = 0.0
        self.accumulated_diff_odd = 0.0
        self.should_calc_even = True
        self.should_calc_odd = True
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("AdaCache state reset")
