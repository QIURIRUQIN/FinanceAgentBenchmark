import asyncio
import itertools
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


DEFAULT_MAX_CONCURRENT = int(os.getenv("KEY_ROTATOR_MAX_CONCURRENT", "50"))


class KeyRotator:
    """Round-robin API key rotator with concurrency control."""

    def __init__(self, keys: list[str], max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> None:
        if not keys:
            raise ValueError("At least one key must be provided")
        self._keys = keys
        self._cycle = itertools.cycle(keys)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @classmethod
    def from_env(cls, env_var: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> "KeyRotator":
        raw = os.getenv(env_var)
        if not raw:
            raise ValueError(f"{env_var} is not set")
        keys = [k.strip() for k in raw.split(";") if k.strip()]
        if not keys:
            raise ValueError(f"{env_var} is empty after parsing")
        return cls(keys, max_concurrent)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[str]:
        """Acquire a semaphore slot and yield the next key. Holds the slot until the caller's async block exits."""
        async with self._semaphore:
            yield next(self._cycle)

    @property
    def key_count(self) -> int:
        return len(self._keys)


class AllKeysExhaustedError(RuntimeError):
    """Raised when every key in a failover pool has reported a usage limit."""


class StickyKeySession:
    """A trajectory-local key lease that changes only after a usage-limit error."""

    def __init__(self, pool: "StickyFailoverKeyPool") -> None:
        self._pool = pool
        self._index: int | None = None

    async def current(self) -> tuple[str, str]:
        if self._index is None:
            self._index = await self._pool.initial_index()
        return self._pool.key_at(self._index), self._pool.label_at(self._index)

    async def failover_after_limit(self) -> tuple[str, str]:
        if self._index is None:
            self._index = await self._pool.initial_index()
        self._index = await self._pool.next_index_after_limit(self._index)
        return self._pool.key_at(self._index), self._pool.label_at(self._index)


class StickyFailoverKeyPool:
    """Shared key pool with trajectory-local stickiness.

    A session keeps its current key indefinitely. Calling
    ``failover_after_limit`` is the only operation that changes it. Exhausted
    keys are remembered globally so newly created trajectories start from the
    first key that is still usable.
    """

    def __init__(self, keys: list[str]) -> None:
        unique_keys = list(dict.fromkeys(key.strip() for key in keys if key.strip()))
        if not unique_keys:
            raise ValueError("At least one key must be provided")
        self._keys = unique_keys
        self._exhausted: set[int] = set()
        self._preferred_index = 0
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(
        cls,
        env_var: str,
        fallback_env_var: str | None = None,
    ) -> "StickyFailoverKeyPool":
        raw = os.getenv(env_var)
        if not raw and fallback_env_var:
            raw = os.getenv(fallback_env_var)
        if not raw:
            fallback_note = f" or {fallback_env_var}" if fallback_env_var else ""
            raise ValueError(f"{env_var}{fallback_note} is not set")
        return cls([key for key in raw.split(";") if key.strip()])

    def session(self) -> StickyKeySession:
        return StickyKeySession(self)

    def key_at(self, index: int) -> str:
        return self._keys[index]

    def label_at(self, index: int) -> str:
        key = self._keys[index]
        return f"{key[:13]}…{key[-6:]}"

    async def initial_index(self) -> int:
        async with self._lock:
            for offset in range(len(self._keys)):
                index = (self._preferred_index + offset) % len(self._keys)
                if index not in self._exhausted:
                    return index
        raise AllKeysExhaustedError("All configured Tavily API keys are exhausted")

    async def next_index_after_limit(self, exhausted_index: int) -> int:
        async with self._lock:
            self._exhausted.add(exhausted_index)
            for offset in range(1, len(self._keys) + 1):
                index = (exhausted_index + offset) % len(self._keys)
                if index not in self._exhausted:
                    self._preferred_index = index
                    return index
        raise AllKeysExhaustedError("All configured Tavily API keys are exhausted")

    @property
    def key_count(self) -> int:
        return len(self._keys)


_rotators: dict[str, KeyRotator] = {}
_sticky_pools: dict[str, StickyFailoverKeyPool] = {}


def get_rotator(env_var: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> KeyRotator:
    """Returns a shared KeyRotator per env var name, creating on first access."""
    if env_var not in _rotators:
        _rotators[env_var] = KeyRotator.from_env(env_var, max_concurrent)
    return _rotators[env_var]


def get_sticky_pool(
    env_var: str,
    fallback_env_var: str | None = None,
) -> StickyFailoverKeyPool:
    """Return one shared sticky failover pool for an environment variable."""
    cache_key = f"{env_var}|{fallback_env_var or ''}"
    if cache_key not in _sticky_pools:
        _sticky_pools[cache_key] = StickyFailoverKeyPool.from_env(
            env_var,
            fallback_env_var=fallback_env_var,
        )
    return _sticky_pools[cache_key]
