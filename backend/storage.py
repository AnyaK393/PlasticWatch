"""Crash- and concurrency-safe JSON storage shared by the API and dashboard processes.

- ``locked(path)``      inter-process exclusive lock (fcntl) around read-modify-write
- ``read_json(path)``   tolerant read; a corrupt file is quarantined, never silently overwritten
- ``write_json(path)``  atomic write (temp file + fsync + os.replace)
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # non-POSIX: thread lock only
    fcntl = None

logger = logging.getLogger(__name__)

_thread_locks: dict = {}
_thread_locks_guard = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    with _thread_locks_guard:
        return _thread_locks.setdefault(str(path.resolve()), threading.Lock())


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Exclusive lock for a data file across threads and processes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(path):
        if fcntl is None:
            yield
            return
        with open(path.with_name(path.name + ".lock"), "a+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class CorruptDataError(RuntimeError):
    """Raised when a data file exists but cannot be parsed (it has been quarantined)."""


def read_json(path: Path, default: Any) -> Any:
    """Read JSON, returning ``default`` if the file does not exist.

    A file that exists but is unreadable is renamed to ``<name>.corrupt-<timestamp>``
    and CorruptDataError is raised, so callers never overwrite real data with defaults.
    """
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        quarantine = path.with_name(f"{path.name}.corrupt-{stamp}")
        os.replace(path, quarantine)
        logger.error("Corrupt data file %s quarantined as %s: %s", path, quarantine.name, exc)
        raise CorruptDataError(f"{path.name} was corrupt and has been moved to {quarantine.name}") from exc


def write_json(path: Path, data: Any) -> None:
    """Atomically replace ``path`` with ``data`` serialised as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
