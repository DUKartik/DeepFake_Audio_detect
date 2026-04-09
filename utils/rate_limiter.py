"""
utils/rate_limiter.py — Redis-backed per-sender daily rate limiter.

Key pattern: verivoice:rate:{sender_hash}:{YYYY-MM-DD}
TTL: 86400 seconds (one full calendar day)
Default limit: 10 analyses per sender per day (configurable via RATE_LIMIT_DAILY env)
"""

import datetime
import redis
import structlog
from utils.exceptions import RateLimitExceededError

log = structlog.get_logger(__name__)


class RateLimiter:
    """Redis-backed per-sender rate limiter.

    Uses a daily counter keyed by a SHA-256 sender hash and the current date.
    On first write the TTL is set to 86400 seconds, ensuring the counter
    expires at most one day after the first request.

    Args:
        redis_url:   Redis connection URL (e.g. "redis://localhost:6379/0").
        daily_limit: Maximum number of requests allowed per sender per day.

    Example:
        limiter = RateLimiter(redis_url="redis://localhost:6379/0", daily_limit=10)
        if not limiter.check("abc123hash"):
            raise RateLimitExceededError(...)
        limiter.increment("abc123hash")
    """

    def __init__(self, redis_url: str, daily_limit: int = 10) -> None:
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self.daily_limit = daily_limit

    def _key(self, sender_hash: str) -> str:
        """Build the Redis key for today's counter for the given sender."""
        today = datetime.date.today().strftime("%Y-%m-%d")
        return f"verivoice:rate:{sender_hash}:{today}"

    def check(self, sender_hash: str) -> bool:
        """Return True if the sender is BELOW the daily limit, False if blocked.

        Args:
            sender_hash: SHA-256 hash of the sender's phone number.

        Returns:
            True  → request is allowed.
            False → sender has exceeded today's quota.
        """
        key = self._key(sender_hash)
        count_str = self._redis.get(key)
        count = int(count_str) if count_str else 0
        allowed = count < self.daily_limit
        log.debug("rate_limit_check", sender=sender_hash[:8], count=count, allowed=allowed)
        return allowed

    def increment(self, sender_hash: str) -> int:
        """Increment the daily counter and set TTL on first write.

        Args:
            sender_hash: SHA-256 hash of the sender's phone number.

        Returns:
            The new counter value after incrementing.
        """
        key = self._key(sender_hash)
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 86400)  # reset expires at most one day after first request
        results = pipe.execute()
        new_count = results[0]
        log.info("rate_limit_increment", sender=sender_hash[:8], new_count=new_count)
        return new_count

    def get_count(self, sender_hash: str) -> int:
        """Return the current request count for the sender today.

        Args:
            sender_hash: SHA-256 hash of the sender's phone number.

        Returns:
            Integer count (0 if no requests made today).
        """
        key = self._key(sender_hash)
        count_str = self._redis.get(key)
        return int(count_str) if count_str else 0
