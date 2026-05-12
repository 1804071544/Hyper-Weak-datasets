"""Progress bar helper with an optional tqdm dependency."""

from __future__ import annotations


def progress(iterable, *, desc=None, total=None, unit=None, leave=False, enabled=True):
    if not enabled:
        return iterable

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable

    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)
