from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_run_id(fmt: str = "%Y%m%dT%H%M%SZ") -> str:
    return utc_now().strftime(fmt)

