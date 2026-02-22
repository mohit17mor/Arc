"""
Logging Middleware — logs all events to file and optionally console.
"""

from __future__ import annotations

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from arc.core.events import Event
from arc.middleware.base import MiddlewareNext


def setup_logging(
    log_dir: Path | None = None,
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Setup Arc logging.
    
    Args:
        log_dir: Directory for log files (default: ~/.arc/logs)
        console_level: Minimum level for console output
        file_level: Minimum level for file output
    
    Returns:
        The configured logger
    """
    log_dir = log_dir or (Path.home() / ".arc" / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("arc")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler (minimal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (detailed output)
    log_file = log_dir / f"arc_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. File: {log_file}")
    
    return logger


class EventLogger:
    """
    Logs all events passing through the system.
    
    Usage:
        event_logger = EventLogger(log_dir=Path("~/.arc/logs"))
        kernel.use(event_logger.middleware)
    """
    
    def __init__(
        self,
        log_dir: Path | None = None,
        log_events: bool = True,
    ) -> None:
        self._log_dir = (log_dir or Path.home() / ".arc" / "logs").expanduser()
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_events = log_events
        
        # Events log file (JSON lines format)
        self._events_file = self._log_dir / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        self._logger = logging.getLogger("arc.events")
    
    async def middleware(self, event: Event, next_handler: MiddlewareNext) -> Event:
        """Log events passing through."""
        
        # Log to Python logger
        self._logger.debug(
            f"[{event.type}] source={event.source} "
            f"data_keys={list(event.data.keys()) if event.data else []}"
        )
        
        # Log to events file (JSON lines)
        if self._log_events:
            self._write_event(event)
        
        return await next_handler(event)
    
    def _write_event(self, event: Event) -> None:
        """Write event to JSON lines file."""
        try:
            record = {
                "timestamp": datetime.now().isoformat(),
                "id": event.id,
                "type": event.type,
                "source": event.source,
                "data": self._safe_serialize(event.data),
                "parent_id": event.parent_id,
            }
            
            with open(self._events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
                
        except Exception as e:
            self._logger.warning(f"Failed to write event log: {e}")
    
    @staticmethod
    def _safe_serialize(data: dict) -> dict:
        """Safely serialize data for JSON."""
        result = {}
        for key, value in data.items():
            try:
                # Test if serializable
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                result[key] = str(value)
        return result