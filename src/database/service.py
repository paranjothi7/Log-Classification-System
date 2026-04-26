"""
database/service.py — CRUD operations and analytics queries.
Supports both SQLite (local dev) and PostgreSQL (production).
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import os
import pandas as pd
from sqlalchemy import func, desc, text
from loguru import logger

from .connection import get_db, engine
from .models import LogEntry, LogCategory, SeverityLevel


def _is_sqlite() -> bool:
    """Check if the current database is SQLite."""
    return "sqlite" in str(engine.url)


class LogService:
    # Write

    @staticmethod
    def save_log(log_data: Dict[str, Any]) -> Optional[LogEntry]:
        try:
            with get_db() as db:
                entry = LogEntry(**log_data)
                db.add(entry)
                db.flush()
                db.refresh(entry)
                return entry
        except Exception as exc:
            logger.error(f"save_log failed: {exc}")
            return None

    @staticmethod
    def bulk_save(logs: List[Dict[str, Any]]) -> int:
        saved = 0
        try:
            with get_db() as db:
                entries = [LogEntry(**d) for d in logs]
                db.bulk_save_objects(entries)
                saved = len(entries)
        except Exception as exc:
            logger.error(f"bulk_save failed: {exc}")
        return saved

    # Read

    @staticmethod
    def get_recent(limit: int = 100,
                   category: Optional[str] = None,
                   severity: Optional[str] = None) -> List[Dict]:
        with get_db() as db:
            q = db.query(LogEntry)
            if category:
                q = q.filter(LogEntry.category == category)
            if severity:
                q = q.filter(LogEntry.severity == severity)
            rows = q.order_by(desc(LogEntry.timestamp)).limit(limit).all()
            return [r.to_dict() for r in rows]

    # Analytics

    @staticmethod
    def category_distribution(days: int = 7) -> pd.DataFrame:
        since = datetime.utcnow() - timedelta(days=days)
        with get_db() as db:
            rows = (
                db.query(
                    LogEntry.category,
                    func.count(LogEntry.id).label("count")
                )
                .filter(LogEntry.timestamp >= since)
                .group_by(LogEntry.category)
                .all()
            )
        return pd.DataFrame(rows, columns=["category", "count"])

    @staticmethod
    def hourly_trend(days: int = 1) -> pd.DataFrame:
        since = datetime.utcnow() - timedelta(days=days)

        try:
            with get_db() as db:
                if _is_sqlite():
                    # ── SQLite: use strftime instead of date_trunc ────────
                    rows = (
                        db.query(
                            func.strftime("%Y-%m-%d %H:00:00",
                                          LogEntry.timestamp).label("hour"),
                            LogEntry.category,
                            func.count(LogEntry.id).label("count"),
                        )
                        .filter(LogEntry.timestamp >= since)
                        .group_by("hour", LogEntry.category)
                        .order_by("hour")
                        .all()
                    )
                else:
                    # PostgreSQL: use date_trunc
                    rows = (
                        db.query(
                            func.date_trunc("hour",
                                            LogEntry.timestamp).label("hour"),
                            LogEntry.category,
                            func.count(LogEntry.id).label("count"),
                        )
                        .filter(LogEntry.timestamp >= since)
                        .group_by("hour", LogEntry.category)
                        .order_by("hour")
                        .all()
                    )

            return pd.DataFrame(rows, columns=["hour", "category", "count"])

        except Exception as exc:
            logger.error(f"hourly_trend failed: {exc}")
            return pd.DataFrame(columns=["hour", "category", "count"])

    @staticmethod
    def summary_stats() -> Dict[str, Any]:
        try:
            with get_db() as db:
                total = db.query(
                    func.count(LogEntry.id)
                ).scalar() or 0

                avg_conf = db.query(
                    func.avg(LogEntry.confidence)
                ).scalar() or 0.0

                critical = db.query(
                    func.count(LogEntry.id)
                ).filter(
                    LogEntry.severity == SeverityLevel.CRITICAL
                ).scalar() or 0

                security = db.query(
                    func.count(LogEntry.id)
                ).filter(
                    LogEntry.category == LogCategory.SECURITY_ALERT
                ).scalar() or 0

            return {
                "total_logs":      total,
                "avg_confidence":  round(float(avg_conf), 3),
                "critical_alerts": critical,
                "security_alerts": security,
            }
        except Exception as exc:
            logger.error(f"summary_stats failed: {exc}")
            return {
                "total_logs":      0,
                "avg_confidence":  0.0,
                "critical_alerts": 0,
                "security_alerts": 0,
            }