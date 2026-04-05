"""
REM Sleep Cycles: Idle-time memory consolidation, compression, and pruning.

Inspired by the biological role of REM sleep in memory consolidation:
- During "REM", the system replays stored episodes.
- Similar memories are merged (consolidation).
- Low-salience memories decay and are pruned.
- Identity-inconsistent memories are down-weighted.
- The HDDR store is compressed via importance-ranked eviction.

A REMProcessor is idle-triggered: it tracks the last activity time and
enters a REM cycle automatically after a configurable idle period.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class REMReport:
    """Summary of a completed REM cycle."""

    cycle_id: int
    start_time: float
    end_time: float
    episodes_consolidated: int
    episodes_pruned: int
    hddr_pruned: int
    identity_alignments_applied: int

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time


class REMProcessor:
    """
    REM Sleep Cycle Processor.

    Performs memory maintenance operations during idle periods:
    1. Decay: reduce importance of all HDDR and episodic memories.
    2. Consolidation: merge similar episodes.
    3. Pruning: remove memories below importance threshold.
    4. Identity alignment: apply identity scoring to boost or down-weight
       memories based on their alignment with the stored identity vector.
    5. HDDR compression: enforce capacity limits.

    The processor can be run manually via ``run_cycle()`` or started as a
    background thread via ``start_background(idle_seconds=...)`` which
    automatically triggers cycles after the specified idle period.
    """

    def __init__(
        self,
        idle_threshold_seconds: float = 30.0,
        decay_rate: float = 0.002,
        min_importance: float = 0.05,
    ) -> None:
        """
        Args:
            idle_threshold_seconds: Seconds of inactivity before auto-REM.
            decay_rate:             Per-cycle importance decay applied to all.
            min_importance:         Prune memories below this importance.
        """
        self.idle_threshold = idle_threshold_seconds
        self.decay_rate = decay_rate
        self.min_importance = min_importance

        self._cycle_count = 0
        self._last_activity = time.time()
        self._reports: List[REMReport] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Hooks — set these to the live memory stores before running
        self.episodic_memory = None   # EpisodicMemory instance
        self.hddr_memory = None       # HDDRMemory instance
        self.identity_anchor = None   # IdentityAnchor instance

        # Optional callback invoked after each cycle
        self.on_cycle_complete: Optional[Callable[[REMReport], None]] = None

    # ------------------------------------------------------------------
    # Activity tracking
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Signal that activity occurred; resets the idle timer."""
        self._last_activity = time.time()

    def is_idle(self) -> bool:
        """Return True if the system has been idle long enough for a REM cycle."""
        return (time.time() - self._last_activity) >= self.idle_threshold

    # ------------------------------------------------------------------
    # Cycle execution
    # ------------------------------------------------------------------

    def run_cycle(self) -> REMReport:
        """
        Execute a single REM cycle synchronously.

        Returns:
            REMReport summarising what was done.
        """
        with self._lock:
            self._cycle_count += 1
            start = time.time()

            consolidated = 0
            ep_pruned = 0
            hddr_pruned = 0
            id_applied = 0

            # --- Step 1: Decay ---
            if self.hddr_memory is not None:
                self.hddr_memory.decay_all()

            if self.episodic_memory is not None:
                for ep in self.episodic_memory.get_all():
                    ep.importance *= (1.0 - self.decay_rate)

            # --- Step 2: Episodic consolidation ---
            if self.episodic_memory is not None:
                consolidated = self.episodic_memory.consolidate()

            # --- Step 3: Identity alignment ---
            if (
                self.episodic_memory is not None
                and self.identity_anchor is not None
                and self.identity_anchor.core is not None
            ):
                for ep in self.episodic_memory.get_all():
                    score = self.identity_anchor.identity_score(ep.embedding)
                    # Boost identity-consistent memories, decay inconsistent ones
                    if score > 0.3:
                        ep.importance = min(ep.importance * 1.1, 10.0)
                    elif score < -0.1:
                        ep.importance *= 0.9
                    id_applied += 1

            # --- Step 4: Pruning ---
            if self.episodic_memory is not None:
                for ep in self.episodic_memory.get_all():
                    ep.importance = max(ep.importance, 0.0)
                ep_pruned = self.episodic_memory.prune()

            if self.hddr_memory is not None:
                hddr_pruned = self.hddr_memory.prune(self.min_importance)

            report = REMReport(
                cycle_id=self._cycle_count,
                start_time=start,
                end_time=time.time(),
                episodes_consolidated=consolidated,
                episodes_pruned=ep_pruned,
                hddr_pruned=hddr_pruned,
                identity_alignments_applied=id_applied,
            )
            self._reports.append(report)

            if self.on_cycle_complete is not None:
                self.on_cycle_complete(report)

            return report

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def start_background(self, poll_interval: float = 5.0) -> None:
        """
        Start a background thread that runs REM cycles when idle.

        Args:
            poll_interval: How often (seconds) to check idle status.
        """
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._background_loop,
            args=(poll_interval,),
            daemon=True,
            name="REMProcessor",
        )
        thread.start()

    def stop_background(self) -> None:
        """Signal the background thread to stop."""
        self._stop_event.set()

    def _background_loop(self, poll_interval: float) -> None:
        while not self._stop_event.is_set():
            time.sleep(poll_interval)
            if self.is_idle():
                self.run_cycle()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def reports(self) -> List[REMReport]:
        return list(self._reports)

    def last_report(self) -> Optional[REMReport]:
        return self._reports[-1] if self._reports else None
