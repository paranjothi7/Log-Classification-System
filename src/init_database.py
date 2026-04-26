#!/usr/bin/env python3
"""
init_database.py
Creates all tables and seeds data from resources/synthetic_logs.csv.
Run once after setting up PostgreSQL (or SQLite for local dev).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from loguru import logger
from pathlib import Path

from database.connection import engine, test_connection
from database.models import Base

CSV_PATH = Path(__file__).parent.parent / "resources" / "synthetic_logs.csv"

# Map CSV target_label → severity (aligned with _severity_for in enhanced_processor)
_SEVERITY_MAP = {
    "Critical Error":      "Critical",
    "Security Alert":      "High",
    "Workflow Error":      "High",
    "Error":               "Medium",
    "Resource Usage":      "Medium",
    "HTTP Status":         "Info",
    "System Notification": "Info",
    "User Action":         "Info",
    "Deprecation Warning": "Low",
    "Unknown":             "Info",
}

# Map CSV complexity column → ClassificationMethod
_METHOD_MAP = {
    "bert":  "BERT",
    "regex": "Regex",
    "llm":   "LLM",
}


def seed_from_csv(limit: int = 500):
    """Load up to `limit` rows from synthetic_logs.csv and write to DB."""
    from database.service import LogService

    if not CSV_PATH.exists():
        logger.warning(f"CSV not found at {CSV_PATH}. Skipping seed.")
        return

    df = pd.read_csv(CSV_PATH).head(limit)
    rows = []
    for _, row in df.iterrows():
        label  = str(row.get("target_label", "Unknown")).strip()
        method = _METHOD_MAP.get(str(row.get("complexity", "bert")).strip().lower(), "BERT")
        rows.append({
            "raw_message":           str(row.get("log_message", "")),
            "source":                str(row.get("source", "unknown")),
            "category":              label,
            "severity":              _SEVERITY_MAP.get(label, "Info"),
            "confidence":            0.90 if method == "Regex" else 0.82,
            "classification_method": method,
            "processing_time_ms":    1.0  if method == "Regex" else 105.0,
        })

    saved = LogService.bulk_save(rows)
    logger.info(f"Seeded {saved} entries from synthetic_logs.csv (limit={limit}).")


def main():
    logger.info("Testing database connection …")
    if not test_connection():
        logger.error("Cannot connect to database. Check DATABASE_URL in .env")
        sys.exit(1)

    logger.info("Creating tables …")
    Base.metadata.create_all(bind=engine)
    logger.success("Tables created.")

    logger.info("Seeding from synthetic_logs.csv …")
    seed_from_csv(limit=500)
    logger.success("Database initialised successfully!")


if __name__ == "__main__":
    main()