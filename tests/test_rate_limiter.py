"""
tests/test_rate_limiter.py — Unit tests for the Redis-backed RateLimiter.

Covers:
  - test_first_request_allowed()
  - test_tenth_request_allowed()
  - test_eleventh_request_blocked()
  - test_counter_resets_after_24h()
  - test_increment_sets_ttl()
  - test_get_count_returns_zero_when_no_requests()

Redis calls are mocked using fakeredis.
"""

import os
import pytest
from datetime import date
from unittest.mock import MagicMock, patch, call

os.environ.setdefault("WA_TOKEN", "t")
os.environ.setdefault("PHONE_NUMBER_ID", "p")
os.environ.setdefault("VERIFY_TOKEN", "v")
os.environ.setdefault("APP_SECRET", "s")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("R2_BUCKET", "b")
os.environ.setdefault("R2_ENDPOINT_URL", "")
os.environ.setdefault("R2_ACCESS_KEY_ID", "")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "")

from utils.rate_limiter import RateLimiter


def _make_limiter(daily_limit: int = 10) -> tuple[RateLimiter, MagicMock]:
    """Create a RateLimiter with a mocked Redis client."""
    with patch("utils.rate_limiter.redis.from_url") as mock_redis_factory:
        mock_redis = MagicMock()
        mock_redis_factory.return_value = mock_redis
        limiter = RateLimiter(redis_url="redis://localhost:6379/0", daily_limit=daily_limit)
    limiter._redis = mock_redis
    return limiter, mock_redis


class TestRateLimiter:
    def test_first_request_allowed(self):
        """A sender with no prior requests today must be allowed."""
        limiter, mock_redis = _make_limiter(daily_limit=10)
        mock_redis.get.return_value = None   # no key in Redis yet
        assert limiter.check("abc123hash") is True

    def test_tenth_request_allowed(self):
        """The 10th request (count = 9 before increment) must still be allowed."""
        limiter, mock_redis = _make_limiter(daily_limit=10)
        mock_redis.get.return_value = "9"    # 9 previous requests
        assert limiter.check("abc123hash") is True

    def test_eleventh_request_blocked(self):
        """The 11th request (count = 10) must be blocked."""
        limiter, mock_redis = _make_limiter(daily_limit=10)
        mock_redis.get.return_value = "10"   # already at daily limit
        assert limiter.check("abc123hash") is False

    def test_increment_returns_new_count(self):
        """increment() must return the incremented count from Redis."""
        limiter, mock_redis = _make_limiter()
        pipe = MagicMock()
        mock_redis.pipeline.return_value = pipe
        pipe.execute.return_value = [5, True]   # INCR returned 5
        count = limiter.increment("abc123hash")
        assert count == 5

    def test_increment_sets_ttl(self):
        """increment() must call EXPIRE with 86400 seconds."""
        limiter, mock_redis = _make_limiter()
        pipe = MagicMock()
        mock_redis.pipeline.return_value = pipe
        pipe.execute.return_value = [1, True]
        limiter.increment("abc123hash")
        pipe.expire.assert_called_once()
        _, args, _ = pipe.expire.mock_calls[0]
        assert args[1] == 86400

    def test_get_count_returns_zero_when_no_requests(self):
        """get_count() must return 0 if no Redis key exists."""
        limiter, mock_redis = _make_limiter()
        mock_redis.get.return_value = None
        assert limiter.get_count("abc123hash") == 0

    def test_get_count_returns_current_value(self):
        """get_count() must return the integer value from Redis."""
        limiter, mock_redis = _make_limiter()
        mock_redis.get.return_value = "7"
        assert limiter.get_count("abc123hash") == 7

    def test_key_includes_today_date(self):
        """The Redis key must embed today's date in YYYY-MM-DD format."""
        limiter, mock_redis = _make_limiter()
        mock_redis.get.return_value = None
        limiter.check("myhash")
        key_used = mock_redis.get.call_args[0][0]
        today_str = date.today().strftime("%Y-%m-%d")
        assert today_str in key_used
        assert "myhash" in key_used

    def test_counter_resets_after_24h(self):
        """Simulates reset by checking that a fresh key (None value) is allowed."""
        limiter, mock_redis = _make_limiter(daily_limit=10)
        # First day: at limit
        mock_redis.get.return_value = "10"
        assert limiter.check("myhash") is False

        # Simulate next day — Redis TTL expired, key is gone
        mock_redis.get.return_value = None
        assert limiter.check("myhash") is True


# ─────────────────────────────────────────────────────────────────────────────
# Hashing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHashing:
    def test_hash_sender_is_sha256(self):
        """hash_sender must return a 64-char hex string (SHA-256)."""
        from utils.hashing import hash_sender
        result = hash_sender("919876543210")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_sender_is_deterministic(self):
        """Same input must always produce the same hash."""
        from utils.hashing import hash_sender
        assert hash_sender("919876543210") == hash_sender("919876543210")

    def test_hash_sender_different_numbers_differ(self):
        """Different numbers must hash to different values."""
        from utils.hashing import hash_sender
        assert hash_sender("919876543210") != hash_sender("919876543211")

    def test_verify_hmac_valid(self):
        """Valid HMAC signature must return True."""
        import hashlib, hmac as _hmac
        from utils.hashing import verify_hmac_sha256
        secret = "mysecret"
        body = b"test payload"
        sig = "sha256=" + _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert verify_hmac_sha256(secret, body, sig) is True

    def test_verify_hmac_invalid(self):
        """Tampered signature must return False."""
        from utils.hashing import verify_hmac_sha256
        assert verify_hmac_sha256("mysecret", b"payload", "sha256=badbadbadbad") is False
