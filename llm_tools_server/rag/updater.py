"""Periodic index updater for long-running RAG applications.

This module provides background sitemap polling and incremental index updates,
enabling RAG indexes to stay current without application restarts.
"""

import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .crawler import SitemapChanges


def _print_flush(msg: str):
    """Print message and flush stdout to ensure visibility in daemon threads."""
    print(msg)
    sys.stdout.flush()


if TYPE_CHECKING:
    from .config import RAGConfig
    from .indexer import DocSearchIndex

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """Result of an incremental index update.

    Attributes:
        success: True if update completed successfully
        pages_added: Number of new pages added to index
        pages_updated: Number of pages re-indexed due to changes
        pages_removed: Number of pages tombstoned/removed
        pages_unchanged: Number of pages that didn't need updating
        chunks_added: Total new chunks added
        chunks_removed: Total chunks tombstoned
        duration_seconds: Time taken for the update
        error: Error message if update failed
        sitemap_changes: The SitemapChanges that triggered this update
        triggered_rebuild: True if update triggered a full index rebuild
    """

    success: bool = True
    pages_added: int = 0
    pages_updated: int = 0
    pages_removed: int = 0
    pages_unchanged: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    sitemap_changes: SitemapChanges | None = None
    triggered_rebuild: bool = False
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "pages_added": self.pages_added,
            "pages_updated": self.pages_updated,
            "pages_removed": self.pages_removed,
            "pages_unchanged": self.pages_unchanged,
            "chunks_added": self.chunks_added,
            "chunks_removed": self.chunks_removed,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "sitemap_changes": self.sitemap_changes.to_dict() if self.sitemap_changes else None,
            "triggered_rebuild": self.triggered_rebuild,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class UpdaterStatus:
    """Current status of the periodic index updater.

    Attributes:
        enabled: True if periodic updates are enabled
        running: True if the background thread is running
        paused: True if updates are temporarily paused (e.g., during user requests)
        last_check: Timestamp of last sitemap check
        last_update: Timestamp of last successful update that found changes
        next_check_eta: Estimated time of next scheduled check
        total_checks: Total number of sitemap checks performed
        total_updates: Total number of updates that found changes
        pages_added_total: Cumulative pages added across all updates
        pages_updated_total: Cumulative pages updated across all updates
        pages_removed_total: Cumulative pages removed across all updates
        tombstone_count: Current number of tombstoned chunks
        tombstone_percentage: Current tombstone percentage
        last_error: Most recent error message (if any)
        errors_count: Total number of errors encountered
    """

    enabled: bool = False
    running: bool = False
    paused: bool = False
    last_check: datetime | None = None
    last_update: datetime | None = None
    next_check_eta: datetime | None = None
    total_checks: int = 0
    total_updates: int = 0
    pages_added_total: int = 0
    pages_updated_total: int = 0
    pages_removed_total: int = 0
    tombstone_count: int = 0
    tombstone_percentage: float = 0.0
    last_error: str | None = None
    errors_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "paused": self.paused,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "next_check_eta": self.next_check_eta.isoformat() if self.next_check_eta else None,
            "total_checks": self.total_checks,
            "total_updates": self.total_updates,
            "pages_added_total": self.pages_added_total,
            "pages_updated_total": self.pages_updated_total,
            "pages_removed_total": self.pages_removed_total,
            "tombstone_count": self.tombstone_count,
            "tombstone_percentage": self.tombstone_percentage,
            "last_error": self.last_error,
            "errors_count": self.errors_count,
        }


class PeriodicIndexUpdater:
    """Background manager for periodic sitemap polling and index updates.

    This class runs a daemon thread that periodically checks the sitemap for
    changes and applies incremental updates to the index. It integrates with
    the existing pause/resume pattern to yield resources during user requests.

    Example usage:
        from llm_tools_server.rag import DocSearchIndex, RAGConfig

        config = RAGConfig(
            base_url="https://docs.example.com",
            periodic_update_enabled=True,
            periodic_update_interval_hours=6.0,
        )

        index = DocSearchIndex(config)
        index.crawl_and_index()
        index.load_index()  # Starts background updater automatically
    """

    def __init__(self, index: "DocSearchIndex", config: "RAGConfig"):
        """Initialize the periodic index updater.

        Args:
            index: The DocSearchIndex to update
            config: RAGConfig with periodic update settings
        """
        self._index = index
        self._config = config

        # Threading controls
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused
        self._updater_thread: threading.Thread | None = None
        self._lock = threading.RLock()

        # Status tracking
        self._last_check: datetime | None = None
        self._last_update: datetime | None = None
        self._total_checks: int = 0
        self._total_updates: int = 0
        self._pages_added_total: int = 0
        self._pages_updated_total: int = 0
        self._pages_removed_total: int = 0
        self._last_error: str | None = None
        self._errors_count: int = 0

        logger.info(
            f"[UPDATER] Initialized with interval={config.periodic_update_interval_hours}h, "
            f"min_interval={config.periodic_update_min_interval_minutes}min, "
            f"batch_size={config.update_batch_size}"
        )

    def start(self):
        """Start the background update thread."""
        with self._lock:
            if self._updater_thread is not None and self._updater_thread.is_alive():
                logger.warning("[UPDATER] Background updater already running")
                return

            self._stop_event.clear()
            self._pause_event.set()  # Ensure unpaused

            self._updater_thread = threading.Thread(
                target=self._run_update_loop,
                name="RAGPeriodicUpdater",
                daemon=True,
            )
            self._updater_thread.start()
            logger.info("[UPDATER] Background updater started")

    def stop(self, timeout: float = 30.0):
        """Stop the background thread gracefully.

        Args:
            timeout: Maximum time to wait for thread to finish
        """
        with self._lock:
            if self._updater_thread is None:
                return

            logger.info("[UPDATER] Stopping background updater...")
            self._stop_event.set()
            self._pause_event.set()  # Unblock if paused

            if self._updater_thread.is_alive():
                self._updater_thread.join(timeout=timeout)
                if self._updater_thread.is_alive():
                    logger.warning(f"[UPDATER] Thread did not stop within {timeout}s timeout")
                else:
                    logger.info("[UPDATER] Background updater stopped")

            self._updater_thread = None

    def pause(self):
        """Pause updates (called during user requests)."""
        self._pause_event.clear()
        logger.debug("[UPDATER] Paused")

    def resume(self):
        """Resume updates (called after user requests)."""
        self._pause_event.set()
        logger.debug("[UPDATER] Resumed")

    def is_running(self) -> bool:
        """Check if the background thread is running."""
        return self._updater_thread is not None and self._updater_thread.is_alive()

    def is_paused(self) -> bool:
        """Check if updates are currently paused."""
        return not self._pause_event.is_set()

    def check_for_updates(self) -> UpdateResult:
        """Check sitemap for changes and apply updates.

        This is the main update logic, called by the background thread
        or manually via force_check().

        Returns:
            UpdateResult with details of what was updated
        """
        import time

        start_time = time.time()
        result = UpdateResult()

        try:
            # Get indexed URLs from the index's crawl state
            indexed_urls = self._index.get_indexed_urls_with_lastmod()

            if not indexed_urls:
                logger.warning("[UPDATER] No indexed URLs found - index may be empty")
                result.error = "No indexed URLs found"
                result.success = False
                return result

            # Get sitemap changes from crawler
            crawler = self._index._create_crawler()
            changes = crawler.get_sitemap_changes(indexed_urls)
            result.sitemap_changes = changes

            if changes.error:
                result.error = changes.error
                result.success = False
                self._last_error = changes.error
                self._errors_count += 1
                return result

            # Track the check
            self._last_check = datetime.now(UTC)
            self._total_checks += 1

            if not changes.has_changes:
                logger.info("[UPDATER] No changes detected in sitemap")
                _print_flush("[UPDATER] No changes detected in sitemap")
                result.pages_unchanged = len(changes.unchanged_urls)
                result.duration_seconds = time.time() - start_time
                return result

            # Apply the changes
            change_msg = (
                f"[UPDATER] Applying changes: {len(changes.new_urls)} new, "
                f"{len(changes.updated_urls)} updated, {len(changes.removed_urls)} removed"
            )
            logger.info(change_msg)
            _print_flush(change_msg)

            # Apply incremental update via the index
            update_result = self._index.apply_incremental_update(changes)

            # Merge results
            result.pages_added = update_result.pages_added
            result.pages_updated = update_result.pages_updated
            result.pages_removed = update_result.pages_removed
            result.pages_unchanged = len(changes.unchanged_urls)
            result.chunks_added = update_result.chunks_added
            result.chunks_removed = update_result.chunks_removed
            result.triggered_rebuild = update_result.triggered_rebuild
            result.success = update_result.success
            result.error = update_result.error

            if result.success:
                self._last_update = datetime.now(UTC)
                self._total_updates += 1
                self._pages_added_total += result.pages_added
                self._pages_updated_total += result.pages_updated
                self._pages_removed_total += result.pages_removed

                success_msg = (
                    f"[UPDATER] Update complete: +{result.pages_added} pages, "
                    f"~{result.pages_updated} updated, -{result.pages_removed} removed, "
                    f"+{result.chunks_added} chunks"
                )
                logger.info(success_msg)
                _print_flush(success_msg)
            else:
                self._last_error = result.error
                self._errors_count += 1
                error_msg = f"[UPDATER] Update failed: {result.error}"
                logger.error(error_msg)
                _print_flush(error_msg)

            result.duration_seconds = time.time() - start_time
            return result

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.duration_seconds = time.time() - start_time
            self._last_error = str(e)
            self._errors_count += 1
            exc_msg = f"[UPDATER] Exception during update: {e}"
            logger.error(exc_msg, exc_info=True)
            _print_flush(exc_msg)
            return result

    def force_check(self) -> UpdateResult:
        """Force an immediate sitemap check, bypassing interval.

        This can be called manually to trigger an update check outside
        the normal schedule.

        Returns:
            UpdateResult with details of what was updated
        """
        logger.info("[UPDATER] Forcing immediate sitemap check")
        return self.check_for_updates()

    def get_status(self) -> UpdaterStatus:
        """Get current updater status for monitoring.

        Returns:
            UpdaterStatus with current state and statistics
        """
        # Calculate next check ETA
        next_check_eta = None
        if self._last_check:
            from datetime import timedelta

            interval = timedelta(hours=self._config.periodic_update_interval_hours)
            next_check_eta = self._last_check + interval

        # Get tombstone info from index
        tombstone_count = 0
        tombstone_percentage = 0.0
        if hasattr(self._index, "_tombstoned_urls"):
            tombstone_count = len(self._index._tombstoned_urls)
            total_chunks = len(self._index._chunks) if hasattr(self._index, "_chunks") else 0
            if total_chunks > 0:
                tombstone_percentage = tombstone_count / total_chunks

        return UpdaterStatus(
            enabled=self._config.periodic_update_enabled,
            running=self.is_running(),
            paused=self.is_paused(),
            last_check=self._last_check,
            last_update=self._last_update,
            next_check_eta=next_check_eta,
            total_checks=self._total_checks,
            total_updates=self._total_updates,
            pages_added_total=self._pages_added_total,
            pages_updated_total=self._pages_updated_total,
            pages_removed_total=self._pages_removed_total,
            tombstone_count=tombstone_count,
            tombstone_percentage=tombstone_percentage,
            last_error=self._last_error,
            errors_count=self._errors_count,
        )

    def _run_update_loop(self):
        """Main background update loop."""
        logger.info("[UPDATER] Background update loop started")
        _print_flush("[UPDATER] Background update loop started")

        # Calculate interval in seconds
        interval_seconds = self._config.periodic_update_interval_hours * 3600
        min_interval_seconds = self._config.periodic_update_min_interval_minutes * 60

        while not self._stop_event.is_set():
            try:
                # Wait for pause to clear (with timeout to check stop event)
                while not self._pause_event.wait(timeout=0.5):
                    if self._stop_event.is_set():
                        logger.info("[UPDATER] Stop requested while paused")
                        return

                if self._stop_event.is_set():
                    break

                # Check if enough time has passed since last check
                current_time = datetime.now(UTC)
                should_check = False

                if self._last_check is None:
                    # First check - wait a bit to let app stabilize
                    logger.info("[UPDATER] First check scheduled in 60 seconds")
                    _print_flush("[UPDATER] First check scheduled in 60 seconds")
                    if self._stop_event.wait(60):
                        break
                    should_check = True
                else:
                    time_since_check = (current_time - self._last_check).total_seconds()
                    if time_since_check >= interval_seconds:
                        should_check = True
                    elif time_since_check < min_interval_seconds:
                        # Too soon - wait for minimum interval
                        wait_time = min_interval_seconds - time_since_check
                        logger.debug(f"[UPDATER] Waiting {wait_time:.0f}s for minimum interval")
                        if self._stop_event.wait(wait_time):
                            break
                        continue

                if should_check:
                    logger.info("[UPDATER] Running scheduled sitemap check")
                    _print_flush("[UPDATER] Running scheduled sitemap check")
                    self.check_for_updates()

                # Sleep until next check (in smaller increments to allow stopping)
                sleep_remaining = interval_seconds
                while sleep_remaining > 0 and not self._stop_event.is_set():
                    sleep_time = min(30, sleep_remaining)  # Check every 30s
                    if self._stop_event.wait(sleep_time):
                        break
                    sleep_remaining -= sleep_time

            except Exception as e:
                logger.error(f"[UPDATER] Error in update loop: {e}", exc_info=True)
                self._last_error = str(e)
                self._errors_count += 1
                # Wait a bit before retrying after error
                if self._stop_event.wait(60):
                    break

        logger.info("[UPDATER] Background update loop exited")
