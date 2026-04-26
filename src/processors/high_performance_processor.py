"""
processors/high_performance_processor.py
Thread-pool based batch processor — 1000+ logs/min.
"""
from __future__ import annotations
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Optional
from dataclasses import dataclass
from loguru import logger

from .enhanced_processor import EnhancedProcessor, ClassificationResult

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "100"))


@dataclass
class BatchResult:
    results:        List[ClassificationResult]
    total:          int
    success:        int
    failed:         int
    duration_sec:   float
    throughput_rpm: float


class HighPerformanceProcessor:
    def __init__(self, max_workers: int = MAX_WORKERS):
        self.max_workers = max_workers
        self._processor  = EnhancedProcessor()

    def process_batch(
        self,
        messages: List[str],
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        t0      = time.perf_counter()
        total   = len(messages)
        results = [None] * total
        failed  = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_map = {
                pool.submit(self._processor.process, msg): idx
                for idx, msg in enumerate(messages)
            }
            done = 0
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error(f"Worker error at index {idx}: {exc}")
                    from .enhanced_processor import ClassificationResult
                    results[idx] = ClassificationResult(
                        category="Unknown", severity="Info",
                        confidence=0.0, method="Hybrid",
                        reasoning=str(exc),
                    )
                    failed += 1
                done += 1
                if progress_cb:
                    progress_cb(done, total)

        duration = time.perf_counter() - t0
        success  = total - failed
        rpm      = round((total / duration) * 60, 1) if duration > 0 else 0.0

        logger.info(
            f"Batch complete: {success}/{total} ok | "
            f"{duration:.2f}s | {rpm} logs/min"
        )
        return BatchResult(
            results=results,
            total=total,
            success=success,
            failed=failed,
            duration_sec=round(duration, 3),
            throughput_rpm=rpm,
        )