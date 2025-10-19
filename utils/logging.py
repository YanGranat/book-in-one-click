#!/usr/bin/env python3
"""
Centralized logging utilities for generation pipelines.

Provides structured, human-readable logging with progress tracking,
stage indicators, and clear status messages for server monitoring.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Optional, Literal
from enum import Enum


class LogLevel(str, Enum):
    """Log severity levels."""
    STAGE = "STAGE"      # Major pipeline stage (e.g., "Structure generation")
    STEP = "STEP"        # Step within a stage (e.g., "Generating section 3/5")
    INFO = "INFO"        # General information
    SUCCESS = "SUCCESS"  # Successful completion
    WARNING = "WARNING"  # Warning message
    ERROR = "ERROR"      # Error message
    DEBUG = "DEBUG"      # Debug information (hidden in production)


class PipelineLogger:
    """
    Structured logger for generation pipelines.
    
    Features:
    - Stage/step hierarchy for clear progress tracking
    - Progress indicators (e.g., [3/5])
    - Timestamps and duration tracking
    - Human-readable messages for monitoring
    - Consistent formatting across all pipelines
    """
    
    def __init__(self, pipeline_name: str, *, show_debug: bool = False):
        """
        Initialize pipeline logger.
        
        Args:
            pipeline_name: Name of the pipeline (e.g., "POST", "ARTICLE", "SERIES")
            show_debug: Whether to show DEBUG level logs
        """
        self.pipeline_name = pipeline_name.upper()
        self.show_debug = show_debug
        self.start_time = time.perf_counter()
        self.stage_start_time: Optional[float] = None
        self.current_stage: Optional[str] = None
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 1:
            return f"{int(seconds * 1000)}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def _log(self, level: LogLevel, message: str, *, progress: Optional[str] = None, extra: Optional[dict] = None):
        """Internal logging method with optional structured key/value payload."""
        if level == LogLevel.DEBUG and not self.show_debug:
            return
        
        elapsed = time.perf_counter() - self.start_time
        timestamp = self._format_duration(elapsed)
        
        # Format progress indicator
        progress_str = f" {progress}" if progress else ""
        
        # Format level with fixed width for alignment
        level_str = f"[{level.value:7s}]"
        
        # Construct message
        kv = ""
        if isinstance(extra, dict) and extra:
            try:
                # Render compact key=value pairs in stable order
                parts = []
                for k in sorted(extra.keys()):
                    try:
                        v = extra[k]
                        if isinstance(v, (dict, list, tuple)):
                            v = str(v)
                        parts.append(f"{k}={v}")
                    except Exception:
                        continue
                if parts:
                    kv = " | " + " ".join(parts)
            except Exception:
                kv = ""
        line = f"[{self.pipeline_name}]{level_str}[{timestamp:>8s}]{progress_str} {message}{kv}"
        
        try:
            print(line, file=sys.stderr, flush=True)
        except Exception:
            pass
    
    def stage(self, stage_name: str, *, total_stages: Optional[int] = None, current_stage: Optional[int] = None):
        """
        Log a major pipeline stage.
        
        Args:
            stage_name: Name of the stage (e.g., "Structure Generation", "Writing Content")
            total_stages: Total number of stages in pipeline
            current_stage: Current stage number (1-based)
        
        Example:
            logger.stage("Structure Generation", total_stages=3, current_stage=1)
            # Output: [POST][STAGE  ][  2.3s] [1/3] Structure Generation
        """
        progress = None
        if total_stages and current_stage:
            progress = f"[{current_stage}/{total_stages}]"
        
        self.current_stage = stage_name
        self.stage_start_time = time.perf_counter()
        
        self._log(LogLevel.STAGE, f"Starting: {stage_name}", progress=progress)
    
    def step(self, step_name: str, *, current: Optional[int] = None, total: Optional[int] = None):
        """
        Log a step within a stage.
        
        Args:
            step_name: Name of the step
            current: Current item number
            total: Total items to process
        
        Example:
            logger.step("Generating subsection", current=3, total=12)
            # Output: [POST][STEP   ][  5.1s] [3/12] Generating subsection
        """
        progress = None
        if current is not None and total is not None:
            progress = f"[{current}/{total}]"
        
        self._log(LogLevel.STEP, step_name, progress=progress)
    
    def info(self, message: str, *, extra: Optional[dict] = None):
        """
        Log general information.
        
        Example:
            logger.info("Using provider: openai, model: gpt-4")
        """
        self._log(LogLevel.INFO, message, extra=extra)
    
    def success(self, message: str, *, show_duration: bool = True, extra: Optional[dict] = None):
        """
        Log successful completion.
        
        Args:
            message: Success message
            show_duration: Whether to append duration of current stage
        
        Example:
            logger.success("Structure generation completed")
            # Output: [POST][SUCCESS][  8.2s] Structure generation completed (took 6.1s)
        """
        if show_duration and self.stage_start_time:
            duration = time.perf_counter() - self.stage_start_time
            message = f"{message} (took {self._format_duration(duration)})"
        
        self._log(LogLevel.SUCCESS, message, extra=extra)
    
    def warning(self, message: str, *, extra: Optional[dict] = None):
        """
        Log warning message.
        
        Example:
            logger.warning("Retrying after transient error")
        """
        self._log(LogLevel.WARNING, message, extra=extra)
    
    def error(self, message: str, *, exception: Optional[Exception] = None, extra: Optional[dict] = None):
        """
        Log error message.
        
        Args:
            message: Error message
            exception: Optional exception to include
        
        Example:
            logger.error("Failed to generate section", exception=e)
        """
        if exception:
            exc_name = type(exception).__name__
            exc_msg = str(exception)[:200]
            message = f"{message}: {exc_name}: {exc_msg}"
        
        self._log(LogLevel.ERROR, message, extra=extra)
    
    def debug(self, message: str, *, extra: Optional[dict] = None):
        """
        Log debug information (only shown if show_debug=True).
        
        Example:
            logger.debug("Attempting retry #2")
        """
        self._log(LogLevel.DEBUG, message, extra=extra)
    
    def retry(self, attempt: int, max_attempts: int, reason: str = ""):
        """
        Log retry attempt.
        
        Args:
            attempt: Current attempt number (1-based)
            max_attempts: Maximum number of attempts
            reason: Optional reason for retry
        
        Example:
            logger.retry(2, 3, reason="API timeout")
            # Output: [POST][WARNING][  3.5s] [2/3] Retry attempt (API timeout)
        """
        progress = f"[{attempt}/{max_attempts}]"
        msg = "Retry attempt"
        if reason:
            msg = f"{msg} ({reason})"
        
        self._log(LogLevel.WARNING, msg, progress=progress)
    
    def parallel_start(self, description: str, total_jobs: int, max_workers: int):
        """
        Log start of parallel execution.
        
        Args:
            description: Description of parallel work
            total_jobs: Total number of jobs
            max_workers: Maximum parallel workers
        
        Example:
            logger.parallel_start("Writing subsections", total_jobs=12, max_workers=4)
            # Output: [POST][INFO   ][  2.1s] Starting parallel: Writing subsections (12 jobs, 4 workers)
        """
        self.info(f"Starting parallel: {description} ({total_jobs} jobs, {max_workers} workers)")
    
    def parallel_complete(self, succeeded: int, failed: int, duration: Optional[float] = None):
        """
        Log completion of parallel execution.
        
        Args:
            succeeded: Number of successful jobs
            failed: Number of failed jobs
            duration: Optional total duration
        
        Example:
            logger.parallel_complete(succeeded=11, failed=1, duration=15.3)
            # Output: [POST][SUCCESS][ 17.4s] Parallel execution complete: 11 succeeded, 1 failed (took 15.3s)
        """
        dur_str = f" (took {self._format_duration(duration)})" if duration else ""
        self.success(f"Parallel execution complete: {succeeded} succeeded, {failed} failed{dur_str}", show_duration=False)
    
    def total_duration(self):
        """
        Log total pipeline duration.
        
        Example:
            logger.total_duration()
            # Output: [POST][SUCCESS][ 45.2s] Pipeline completed in 45.2s
        """
        elapsed = time.perf_counter() - self.start_time
        self.success(f"Pipeline completed in {self._format_duration(elapsed)}", show_duration=False)


def create_logger(pipeline: Literal["post", "article", "series", "summary", "meme"], *, show_debug: bool = False) -> PipelineLogger:
    """
    Create a pipeline logger instance.
    
    Args:
        pipeline: Pipeline type
        show_debug: Whether to show debug logs
    
    Returns:
        Configured PipelineLogger instance
    
    Example:
        logger = create_logger("post")
        logger.stage("Writing", total_stages=3, current_stage=1)
        logger.step("Draft generation")
        logger.success("Post generated successfully")
    """
    return PipelineLogger(pipeline, show_debug=show_debug)

