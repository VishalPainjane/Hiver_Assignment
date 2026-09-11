"""High-performance Redis cache with text normalization, reconnect logic, and local fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger("service.cache")

try:
    import redis
    REDIS_INSTALLED = True
except ImportError:
    REDIS_INSTALLED = False


def normalize_cache_key(text: str) -> str:
    """Normalize whitespace and casing before generating SHA-256 cache key."""
    cleaned = " ".join(text.strip().lower().split())
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


class IntentCache:
    """Thread-safe Redis caching layer with periodic auto-reconnect and LRU fallback."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        max_local_size: int = 10000,
        reconnect_interval_sec: float = 10.0,
    ) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = default_ttl
        self.max_local_size = max_local_size
        self.reconnect_interval = reconnect_interval_sec

        self._redis_client: Optional[Any] = None
        self._redis_available = False
        self._last_reconnect_attempt = 0.0

        # Local LRU fallback: OrderedDict[key, (data, expire_at)]
        self._local_cache: OrderedDict[str, tuple[Dict[str, Any], float]] = OrderedDict()
        self._lock = threading.Lock()

        self._connect_redis()

    def _connect_redis(self) -> bool:
        """Attempt to establish or re-verify Redis connection."""
        now = time.time()
        self._last_reconnect_attempt = now

        if not REDIS_INSTALLED:
            self._redis_available = False
            return False

        try:
            client = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=True,
                retry_on_timeout=True,
            )
            client.ping()
            self._redis_client = client
            self._redis_available = True
            logger.info(f"Connected to Redis cache at {self.redis_url}")
            return True
        except Exception as e:
            self._redis_available = False
            self._redis_client = None
            logger.debug(f"Redis not available ({e}). Using in-process LRU cache fallback.")
            return False

    def _check_reconnect(self) -> None:
        """Attempt to reconnect to Redis if reconnect interval has elapsed."""
        if not self._redis_available:
            now = time.time()
            if now - self._last_reconnect_attempt >= self.reconnect_interval:
                self._connect_redis()

    def is_redis_connected(self) -> bool:
        """Check whether live Redis server is active and responding."""
        if not self._redis_available or self._redis_client is None:
            self._check_reconnect()
            return self._redis_available
        try:
            return bool(self._redis_client.ping())
        except Exception:
            self._redis_available = False
            return False

    def get(self, text: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached classification result for the normalized text."""
        key = f"cache:intent:{normalize_cache_key(text)}"

        # 1. Try Redis
        self._check_reconnect()
        if self._redis_available and self._redis_client is not None:
            try:
                val = self._redis_client.get(key)
                if val is not None:
                    return json.loads(val)
            except Exception as e:
                logger.warning(f"Redis get error: {e}. Falling back to local cache.")
                self._redis_available = False

        # 2. Local LRU fallback
        with self._lock:
            if key in self._local_cache:
                data, expire_at = self._local_cache[key]
                if time.time() < expire_at:
                    self._local_cache.move_to_end(key)
                    return data
                else:
                    del self._local_cache[key]
        return None

    def set(self, text: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Store classification result in cache."""
        key = f"cache:intent:{normalize_cache_key(text)}"
        ttl = ttl or self.default_ttl

        # 1. Store in Redis
        if self._redis_available and self._redis_client is not None:
            try:
                self._redis_client.set(key, json.dumps(data), ex=ttl)
            except Exception as e:
                logger.warning(f"Redis set error: {e}. Falling back to local cache.")
                self._redis_available = False

        # 2. Store in local LRU cache
        with self._lock:
            expire_at = time.time() + ttl
            self._local_cache[key] = (data, expire_at)
            self._local_cache.move_to_end(key)
            if len(self._local_cache) > self.max_local_size:
                self._local_cache.popitem(last=False)
