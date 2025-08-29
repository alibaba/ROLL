import asyncio
import time
from typing import Dict
import ray
from roll.utils.constants import RAY_NAMESPACE

@ray.remote
class GlobalLimiter:
    """Global call rate limiter, controls the concurrent number of all tool calls"""
    
    def __init__(self, max_concurrent_calls: int = 10):
        self.max_concurrent_calls = max_concurrent_calls
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)
        self.active_calls = 0
        self.total_calls = 0
        self.stats = {
            "total_calls": 0,
            "active_calls": 0,
            "max_active_calls": 0,
            "waiting_count": 0
        }

    async def acquire(self) -> str:
        """Acquire call permission"""
        self.stats["waiting_count"] += 1
        acquire_id = f"{time.time()}_{self.total_calls}"
        
        await self.semaphore.acquire()
        self.active_calls += 1
        self.total_calls += 1
        self.stats["active_calls"] = self.active_calls
        self.stats["total_calls"] = self.total_calls
        self.stats["max_active_calls"] = max(self.stats["max_active_calls"], self.active_calls)
        self.stats["waiting_count"] -= 1
        
        return acquire_id
    
    async def release(self, acquire_id: str):
        """Release call permission"""
        self.active_calls -= 1
        self.stats["active_calls"] = self.active_calls
        self.semaphore.release()

    async def get_stats(self) -> Dict:
        """Get statistics information"""
        return self.stats.copy()
    
    async def update_limit(self, new_limit: int):
        """Dynamically update concurrent limit"""
        if new_limit > 0:
            old_limit = self.max_concurrent_calls
            self.max_concurrent_calls = new_limit
            
            # Adjust semaphore
            if new_limit > old_limit:
                # Add permits
                for _ in range(new_limit - old_limit):
                    self.semaphore.release()
            elif new_limit < old_limit:
                # Reduce permits (by acquiring excess permits)
                for _ in range(old_limit - new_limit):
                    await self.semaphore.acquire()


class LimiterClient:
    """Rate limiter client, provides synchronous interface"""
    
    def __init__(self, tag: str = "default", max_concurrent_calls: int = 10):
        self.tag = tag
        self.limiter = None
        self.max_concurrent_calls = max_concurrent_calls
        self._initialize_limiter()
    
    def _initialize_limiter(self):
        """Initialize global rate limiter"""
        limiter_name = f"GlobalLimiter_{self.tag}"
        self.limiter = GlobalLimiter.options(
            name=limiter_name,
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote(max_concurrent_calls=self.max_concurrent_calls)

    def acquire(self) -> str:
        """Synchronously acquire call permission"""
        if self.limiter is None:
            self._initialize_limiter()
        return ray.get(self.limiter.acquire.remote())
    
    def release(self, acquire_id: str):
        """Synchronously release call permission"""
        if self.limiter is None:
            self._initialize_limiter()
        ray.get(self.limiter.release.remote(acquire_id))
    
    def get_stats(self) -> Dict:
        """Get statistics information"""
        if self.limiter is None:
            self._initialize_limiter()
        return ray.get(self.limiter.get_stats.remote())
    
    def update_limit(self, new_limit: int):
        """Update concurrent limit"""
        if self.limiter is None:
            self._initialize_limiter()
        ray.get(self.limiter.update_limit.remote(new_limit))

    def __enter__(self):
        self._acquire_id = self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release(self._acquire_id)

# Global singleton instances
_global_limiters = {}

def get_global_limiter(tag: str = "default", max_concurrent_calls: int = 10) -> LimiterClient:
    """Get API rate limiter instance for specified tag"""
    global _global_limiters
    if tag not in _global_limiters:
        _global_limiters[tag] = LimiterClient(tag=tag, max_concurrent_calls=max_concurrent_calls)
    return _global_limiters[tag]

def clear_global_limiters(tag: str = None):
    """Clear limiter instances
    
    Args:
        tag: The tag to clear, if None then clear all instances
    """
    global _global_limiters
    if tag is None:
        _global_limiters.clear()
    elif tag in _global_limiters:
        del _global_limiters[tag]

def get_active_limiter_tags() -> list:
    """Get list of all active limiter tags"""
    global _global_limiters
    return list(_global_limiters.keys()) 