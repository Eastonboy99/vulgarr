"""Bounded classifier transport. Never interpret failure as negative evidence."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
import weakref


class WorkerBudgetExceeded(RuntimeError):
    pass


class WorkersUnavailable(RuntimeError):
    pass


class SourceScanUnavailable(WorkersUnavailable):
    """A source/chunk failed, not evidence of a poisoned inference worker."""


@dataclass
class ScanBudget:
    max_failures: int = 12
    max_cpu_frames: int = 64
    cpu_seconds: float = 30.
    failures: int = 0
    cpu_frames: int = 0
    cpu_elapsed: float = 0
    exhausted: bool = False
    source_disabled: set = field(default_factory=set)
    _cpu_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def check(self):
        if self.exhausted or self.failures >= self.max_failures:
            raise WorkerBudgetExceeded('Classifier retry/fallback budget exhausted; no partial scan activated') from None

    def failed(self):
        self.failures += 1
        self.check()

    async def local(self, call, path):
        async with self._cpu_lock:
            self.check()
            now = time.monotonic()
            remaining = self.cpu_seconds-self.cpu_elapsed
            if self.cpu_frames >= self.max_cpu_frames or remaining <= 0:
                self.exhausted = True
                self.check()
            self.cpu_frames += 1
            try:
                return await asyncio.wait_for(asyncio.to_thread(call, path), timeout=remaining)
            except TimeoutError:
                # An in-progress native call cannot be forcibly killed in a
                # thread. Admit no more CPU work after this deadline.
                self.exhausted = True
                raise WorkerBudgetExceeded('Classifier CPU fallback deadline exceeded; scan not activated') from None
            finally:
                self.cpu_elapsed += time.monotonic()-now


_budget = ContextVar('classifier_budget', default=None)


@contextmanager
def scan_budget():
    existing = _budget.get()
    token = _budget.set(existing or ScanBudget())
    try:
        yield _budget.get()
    finally:
        _budget.reset(token)


def budget():
    return _budget.get()


class WorkerPool:
    def __init__(self, cooldown=300., clock=time.monotonic):
        self.cooldown, self.clock = cooldown, clock
        self.quarantined = {}
        self._locks = weakref.WeakKeyDictionary()
        self.cursor = 0

    def _lock(self, url):
        locks = self._locks.setdefault(asyncio.get_running_loop(), {})
        return locks.setdefault(url, asyncio.Lock())

    def blocked_count(self, urls):
        return sum(self.quarantined.get(url, 0) > self.clock() for url in urls)

    async def request(self, urls, operation, *, source=False):
        if not urls:
            raise WorkersUnavailable('No remote classifiers configured')
        start = self.cursor % len(urls)
        self.cursor += 1
        for url in urls[start:]+urls[:start]:
            async with self._lock(url):
                if self.quarantined.get(url, 0) > self.clock():
                    continue
                if source and budget() and url in budget().source_disabled:
                    continue
                if budget(): budget().check()
                try:
                    result = await operation(url)
                except asyncio.CancelledError:
                    raise
                except WorkerBudgetExceeded:
                    raise
                except SourceScanUnavailable:
                    # Do not poison all titles because one movie/HTTP source
                    # failed. Disable this extraction route for this scan only;
                    # healthy per-frame inference can still be used.
                    if budget():
                        budget().source_disabled.add(url)
                        budget().failed()
                    continue
                except Exception:
                    # No URL, response payload, token or exception text escapes.
                    self.quarantined[url] = self.clock()+self.cooldown
                    if budget(): budget().failed()
                    continue
                self.quarantined.pop(url, None)
                return result
        raise WorkersUnavailable('Remote classifiers quarantined/unavailable; bounded fallback only')


pool = WorkerPool()
