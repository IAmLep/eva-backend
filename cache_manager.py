"""
Cache Manager module for EVA backend.

This module provides in-memory caching functionalities for 
performance optimization. Note that Redis has been removed in v1.8.1.

"""

"""
Version 3 working
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

from config import get_settings

# Logger configuration
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class CacheStrategy(str, Enum):
    """
    Cache strategy options.
    
    Attributes:
        MEMORY: In-memory caching (default)
        NONE: No caching
    """
    MEMORY = "memory"
    NONE = "none"


class CacheManager:
    """
    Cache Manager for handling in-memory caching.
    
    This class provides a simple in-memory cache with TTL support.
    It was refactored from the previous Redis implementation.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of items in cache
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._max_size = max_size
        self._strategy = CacheStrategy.MEMORY
        self._hits = 0
        self._misses = 0
        logger.info(f"Cache manager initialized with strategy: {self._strategy}")
    
    @property
    def strategy(self) -> CacheStrategy:
        """
        Get current cache strategy.
        
        Returns:
            CacheStrategy: Current strategy
        """
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: CacheStrategy):
        """
        Set cache strategy.
        
        Args:
            strategy: Cache strategy to use
        """
        self._strategy = strategy
        logger.info(f"Cache strategy set to: {strategy}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value if found and not expired, None otherwise
        """
        if self._strategy == CacheStrategy.NONE:
            self._misses += 1
            return None
        
        if key not in self._cache:
            self._misses += 1
            return None
        
        value, expire_at = self._cache[key]
        
        # Check if expired
        if expire_at < time.time():
            self._remove(key)
            self._misses += 1
            return None
        
        self._hits += 1
        return value
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            
        Returns:
            bool: True if successful
        """
        if self._strategy == CacheStrategy.NONE:
            return False
        
        # Check if cache is full
        if len(self._cache) >= self._max_size and key not in self._cache:
            self._evict_oldest()
        
        # Calculate expiration time
        expire_at = time.time() + ttl
        
        # Store value
        self._cache[key] = (value, expire_at)
        return True
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if key was found and deleted
        """
        return self._remove(key)
    
    def flush_all(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            bool: True if successful
        """
        self._cache.clear()
        logger.info("Cache flushed")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        return {
            "strategy": self._strategy,
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_ratio": self._calculate_hit_ratio(),
            "hits": self._hits,
            "misses": self._misses,
            "keys": list(self._cache.keys()) if len(self._cache) < 100 else f"{len(self._cache)} keys"
        }
    
    def prune_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            int: Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, (_, expire_at) in self._cache.items() 
            if expire_at < now
        ]
        
        for key in expired_keys:
            self._remove(key)
        
        count = len(expired_keys)
        if count > 0:
            logger.info(f"Pruned {count} expired cache entries")
        
        return count
    
    def _remove(self, key: str) -> bool:
        """
        Remove a key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if key was found and removed
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def _evict_oldest(self) -> bool:
        """
        Evict oldest entry from cache.
        
        Returns:
            bool: True if successful
        """
        if not self._cache:
            return False
        
        # Find key with earliest expiration
        oldest_key = min(
            self._cache.items(),
            key=lambda item: item[1][1]  # Sort by expiration time
        )[0]
        
        return self._remove(oldest_key)
    
    def _calculate_hit_ratio(self) -> float:
        """
        Calculate cache hit ratio.
        
        Returns:
            float: Hit ratio (0.0 to 1.0)
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total


# Singleton instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """
    Get cache manager singleton.
    
    Returns:
        CacheManager: Cache manager instance
    """
    global _cache_manager
    if _cache_manager is None:
        settings = get_settings()
        max_size = getattr(settings, "CACHE_MAX_SIZE", 1000)
        _cache_manager = CacheManager(max_size=max_size)
    return _cache_manager


def cached(
    ttl: int = 300,
    key_prefix: str = "",
    key_builder: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache key
        key_builder: Optional function to build cache key
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache_manager()
            
            # Skip cache if strategy is NONE
            if cache.strategy == CacheStrategy.NONE:
                return await func(*args, **kwargs)
            
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key: prefix:func_name:args:kwargs
                args_str = ":".join(str(arg) for arg in args)
                kwargs_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{args_str}:{kwargs_str}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache_manager()
            
            # Skip cache if strategy is NONE
            if cache.strategy == CacheStrategy.NONE:
                return func(*args, **kwargs)
            
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key: prefix:func_name:args:kwargs
                args_str = ":".join(str(arg) for arg in args)
                kwargs_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{args_str}:{kwargs_str}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, ttl)
            
            return result
        
        # Determine if function is async or sync
        if asyncio_is_coroutine_function(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)
    
    return decorator


def invalidate_cache(
    key_or_pattern: str,
    is_pattern: bool = False
) -> bool:
    """
    Invalidate cache entries.
    
    Args:
        key_or_pattern: Cache key or pattern
        is_pattern: Whether the key is a pattern
        
    Returns:
        bool: True if successful
    """
    cache = get_cache_manager()
    
    if is_pattern:
        # Simple pattern matching (exact prefix match)
        keys_to_delete = [
            key for key in cache._cache.keys()
            if key.startswith(key_or_pattern)
        ]
        
        for key in keys_to_delete:
            cache.delete(key)
        
        if keys_to_delete:
            logger.info(f"Invalidated {len(keys_to_delete)} cache entries with pattern '{key_or_pattern}'")
        
        return bool(keys_to_delete)
    else:
        result = cache.delete(key_or_pattern)
        if result:
            logger.info(f"Invalidated cache entry: {key_or_pattern}")
        return result


def asyncio_is_coroutine_function(func: Callable) -> bool:
    """
    Check if a function is an asyncio coroutine function.
    
    Args:
        func: Function to check
        
    Returns:
        bool: True if function is a coroutine function
    """
    # Import here to avoid circular imports
    import asyncio
    import inspect
    
    return (
        asyncio.iscoroutinefunction(func) or 
        inspect.isgeneratorfunction(func) and any(
            "await" in line or "yield" in line
            for line in inspect.getsource(func).splitlines()
        )
    )


async def start_cache_maintenance_task():
    """
    Start periodic cache maintenance tasks.
    
    This function starts a background task to prune expired entries.
    In a production environment, this should be called on application startup.
    """
    import asyncio
    
    cache = get_cache_manager()
    
    async def maintenance_loop():
        while True:
            try:
                # Prune expired entries
                cache.prune_expired()
                
                # Report stats if significant activity
                if cache._hits + cache._misses > 1000:
                    stats = cache.get_stats()
                    logger.info(f"Cache stats: {stats['size']}/{stats['max_size']} entries, "
                               f"hit ratio: {stats['hit_ratio']:.2f}")
                
                # Wait for next run
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"Error in cache maintenance task: {str(e)}")
                await asyncio.sleep(60)  # Shorter interval on error
    
    # Start task
    task = asyncio.create_task(maintenance_loop())
    
    # Store task reference to prevent garbage collection
    cache._maintenance_task = task
    
    logger.info("Cache maintenance task started")