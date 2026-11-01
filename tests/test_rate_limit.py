"""Tests for the in-memory sliding window rate limiter."""

import pytest

from docinsight.core.rate_limit import InMemoryRateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def limiter(clock: FakeClock) -> InMemoryRateLimiter:
    return InMemoryRateLimiter(window_s=60, clock=clock)


def test_requests_up_to_the_limit_are_allowed(limiter: InMemoryRateLimiter) -> None:
    assert [limiter.hit("a", 3) for _ in range(3)] == [None, None, None]


def test_the_request_over_the_limit_gets_the_wait_time(
    limiter: InMemoryRateLimiter, clock: FakeClock
) -> None:
    limiter.hit("a", 2)
    clock.now += 10
    limiter.hit("a", 2)
    clock.now += 5

    # The first request leaves the window 60 s after it was made, 45 s from now.
    assert limiter.hit("a", 2) == 45


def test_the_window_slides(limiter: InMemoryRateLimiter, clock: FakeClock) -> None:
    limiter.hit("a", 2)
    clock.now += 30
    limiter.hit("a", 2)
    clock.now += 30  # the first request is now exactly 60 s old: outside the window

    assert limiter.hit("a", 2) is None
    assert limiter.hit("a", 2) is not None  # the second one is still inside


def test_rejected_requests_are_not_counted(limiter: InMemoryRateLimiter, clock: FakeClock) -> None:
    limiter.hit("a", 1)
    for _ in range(5):
        clock.now += 10
        assert limiter.hit("a", 1) is not None
    clock.now += 10  # 60 s after the only counted request

    assert limiter.hit("a", 1) is None


def test_keys_are_counted_separately(limiter: InMemoryRateLimiter) -> None:
    limiter.hit("a", 1)

    assert limiter.hit("a", 1) is not None
    assert limiter.hit("b", 1) is None


def test_old_keys_are_swept(limiter: InMemoryRateLimiter, clock: FakeClock) -> None:
    limiter.hit("old", 5)
    clock.now += 61
    for i in range(InMemoryRateLimiter.SWEEP_EVERY - 1):
        limiter.hit(f"new-{i % 10}", 10_000)

    assert "old" not in limiter._hits
    assert len(limiter._hits) == 10
