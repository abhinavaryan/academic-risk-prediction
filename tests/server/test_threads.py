"""Native thread pools must be limited inside the worker threads that run requests."""

import threading

from threadpoolctl import threadpool_info, threadpool_limits

from riskml.server.main import native_thread_limit


def _max_threads():
    return max((lib["num_threads"] for lib in threadpool_info()), default=1)


def test_limit_applies_in_worker_thread():
    seen = {}

    def worker():
        threadpool_limits(limits=3)  # a fresh worker thread on a multi-core host
        seen["before"] = _max_threads()
        with native_thread_limit():
            seen["inside"] = _max_threads()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["before"] > 1
    assert seen["inside"] == 1
