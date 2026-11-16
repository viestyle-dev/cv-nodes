"""Background worker that keeps only the newest pending job (latest wins).

Heavy OpenCV work must not run on the event loop thread, because the same
thread also produces audio blocks and serves the engine. A node submits jobs
from its input callback; the worker runs them one at a time on its own thread
and hands each result back to the event loop with ``call_soon_threadsafe``.
A job submitted while another is still waiting replaces it, so a slow worker
never builds a backlog.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class LatestWinsWorker:
    """Run ``work(job)`` on a background thread; deliver results on the loop.

    ``on_result(result)`` is called on the event loop thread for every job
    whose ``work`` returned something other than ``None``. Because the
    dispatch it performs starts outside any port callback, ``on_result`` is
    responsible for flushing the graph's outputs.
    """

    def __init__(
        self,
        name: str,
        work: Callable[[Any], Any],
        on_result: Callable[[Any], None],
    ):
        self._name = name
        self._work = work
        self._on_result = on_result

        self._cond = threading.Condition()
        self._pending: Any = None
        self._has_pending = False
        self._stopped = True

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        #: Number of jobs replaced by a newer one before they ran.
        self.dropped = 0

    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._thread is not None:
            return
        self._loop = loop
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run, name=f"{self._name}-worker", daemon=True
        )
        self._thread.start()

    def submit(self, job: Any) -> None:
        with self._cond:
            if self._has_pending:
                self.dropped += 1
            self._pending = job
            self._has_pending = True
            self._cond.notify()

    def stop(self, timeout: float = 2.0) -> None:
        thread = self._thread
        if thread is None:
            return
        with self._cond:
            self._stopped = True
            self._pending = None
            self._has_pending = False
            self._cond.notify()
        thread.join(timeout)
        self._thread = None
        self._loop = None

    # -- worker thread -------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._has_pending and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                job = self._pending
                self._pending = None
                self._has_pending = False

            try:
                result = self._work(job)
            except Exception:
                logger.exception("%s: worker job failed", self._name)
                continue

            loop = self._loop
            if result is None or loop is None or self._stopped:
                continue
            try:
                loop.call_soon_threadsafe(self._deliver, result)
            except RuntimeError:
                # The event loop is closed; nothing left to deliver to.
                return

    # -- event loop thread ---------------------------------------------------

    def _deliver(self, result: Any) -> None:
        if self._stopped:
            return
        try:
            self._on_result(result)
        except Exception:
            logger.exception("%s: delivering a worker result failed", self._name)


class ThreadedComputeMixin:
    """Mixin for nodes that offload ``_compute(job)`` to a LatestWinsWorker.

    The node implements ``_compute(job) -> result | None`` (pure, may run on
    the worker thread) and ``_emit(result)`` (dispatches outputs, runs on the
    event loop). With the ``threaded`` property off, or before activation
    (tests, graph construction), jobs run inline in the caller's dispatch and
    the caller keeps the flush responsibility.
    """

    _worker: LatestWinsWorker

    def _init_worker(self) -> None:
        self._worker = LatestWinsWorker(
            self.name, self._compute, self._deliver_result  # type: ignore[attr-defined]
        )

    def _sync_worker(self) -> None:
        """Start or stop the worker to match the ``threaded`` property."""
        want = bool(self.properties.get("threaded", True)) and self.is_active  # type: ignore[attr-defined]
        if want and not self._worker.running:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._worker.start(loop)
        elif not want and self._worker.running:
            self._worker.stop()

    def _submit(self, job: Any) -> None:
        if self._worker.running:
            self._worker.submit(job)
            return
        result = self._compute(job)  # type: ignore[attr-defined]
        if result is not None:
            self._emit(result)  # type: ignore[attr-defined]

    def _deliver_result(self, result: Any) -> None:
        self._emit(result)  # type: ignore[attr-defined]
        graph = self.parent_graph  # type: ignore[attr-defined]
        if graph is not None:
            graph.flush_outputs()
