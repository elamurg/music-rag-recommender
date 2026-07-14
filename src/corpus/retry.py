"""wraps functions that make HTTP calls with exponential backoff on transient
errors. Permanent errors (item not found, bad credentials) are re-raised immediately so callers can skip
and move on without wasting retries."""
import functools
import time
from typing import Callable, TypeVar

import pylast
import requests

T = TypeVar("T")

# retry configuration 
MAX_RETRIES = 3
INITIAL_BACKOFF_SEC = 5
BACKOFF_MULTIPLIER = 3  

PERMANENT_ERROR_MARKERS = (
    "track not found",
    "artist not found",
    "album not found",
    "invalid parameters",
    "invalid api key",
)


def is_permanent_pylast_error(exc: Exception) -> bool:
    """Classify pylast WSErrors as permanent (skip) vs transient (retry)."""
    if not isinstance(exc, pylast.WSError):
        return False
    details = str(exc.details).lower() if exc.details else ""
    return any(marker in details for marker in PERMANENT_ERROR_MARKERS)


def retry(fn: Callable[..., T]) -> Callable[..., T]:
    """Retry with exponential backoff on failures"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        backoff = INITIAL_BACKOFF_SEC
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except (requests.RequestException, TimeoutError, OSError,ConnectionError) as e:
                last_exc = e
            except pylast.WSError as e:
                if is_permanent_pylast_error(e):
                    raise
                last_exc = e

            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
        raise last_exc  
    return wrapper

