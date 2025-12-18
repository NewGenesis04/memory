"""
A simple in-memory cache with Time-To-Live (TTL) and max size.
"""

import time
from collections import OrderedDict
from typing import Any, Optional

class Cache:
    """
    A simple in-memory LRU (Least Recently Used) cache with TTL.
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initializes the cache.
        
        Args:
            max_size: The maximum number of items to store in the cache.
            ttl_seconds: The time-to-live for each cache item, in seconds.
        """
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache = OrderedDict() # Stores {key: (value, timestamp)}

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache.
        
        Returns the item if it exists and has not expired, otherwise None.
        """
        if key not in self._cache:
            return None # Cache miss

        value, timestamp = self._cache[key]

        # Check if TTL has expired
        if time.time() - timestamp > self.ttl:
            # Item has expired, remove it
            del self._cache[key]
            return None # Cache miss

        # Move item to the end to mark it as recently used
        self._cache.move_to_end(key)
        
        return value # Cache hit

    def set(self, key: str, value: Any):
        """
        Adds an item to the cache.
        
        If the cache is full, the least recently used item is removed.
        """
        if key in self._cache:
            # If key already exists, just update it and move to end
            self._cache.move_to_end(key)
        
        self._cache[key] = (value, time.time())

        # Enforce max size by removing the least recently used item
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False) # popitem(last=False) acts as FIFO for OrderedDict

    def clear(self):
        """Clears the entire cache."""
        self._cache.clear()

    def __len__(self):
        return len(self._cache)

__all__ = ["Cache"]
