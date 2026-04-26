"""
src/run_pipeline.py
Run the full classification pipeline on synthetic_logs.csv
and save results to CSV + JSON + Database.
"""
import sys, os

# Fix path so processors/database/utils are always found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import pandas as pd
from pathlib import Path
from loguru import logger
from processors.high_performance_processor import HighPerformanceProcessor
from utils.result_saver import ResultSaver

CSV_PATH = Path(BASE_DIR).parent / "resources" / "synthetic_logs.csv"


def main():
    # Verify CSV exists
    if not CSV_PATH.exists():
        logger.error(f"CSV not found at: {CSV_PATH}")
        logger.error("Make sure synthetic_logs.csv is in the resources/ folder.")
        sys.exit(1)

    # Load data
    logger.info(f"Loading logs from {CSV_PATH} ...")
    df       = pd.read_csv(CSV_PATH)
    messages = df["log_message"].astype(str).tolist()
    sources  = df["source"].astype(str).tolist()
    labels   = df["target_label"].astype(str).tolist()
    logger.info(f"Loaded {len(messages)} log messages.")

    # Classify
    logger.info("Starting classification pipeline ...")
    processor = HighPerformanceProcessor(max_workers=4)

    def progress(done, total):
        if done % 100 == 0 or done == total:
            logger.info(f"  Progress: {done}/{total}")

    batch = processor.process_batch(messages, progress_cb=progress)
    logger.success(
        f"Done — {batch.success}/{batch.total} classified in "
        f"{batch.duration_sec:.1f}s ({batch.throughput_rpm} logs/min)"
    )

    # Build results rows
    results = []
    for i, r in enumerate(batch.results):
        results.append({
            "timestamp":             df["timestamp"].iloc[i],
            "source":                sources[i],
            "log_message":           messages[i],
            "ground_truth":          labels[i],
            "category":              r.category,
            "severity":              r.severity,
            "confidence":            round(r.confidence, 4),
            "classification_method": r.method,
            "processing_time_ms":    r.processing_time_ms,
            "correct":               labels[i] == r.category,
        })

    # Save results
    saver = ResultSaver()

    csv_path  = saver.to_csv(results,  "classified_synthetic_logs.csv")
    json_path = saver.to_json(results, "classified_synthetic_logs.json")

    # Optional DB save — skip if DB not configured
    try:
        db_saved = saver.to_database(results)
    except Exception as exc:
        logger.warning(f"DB save skipped: {exc}")
        db_saved = 0

    # Summary
    saver.summary_report(results)

    correct  = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) * 100

    print(f"""

              PIPELINE RESULTS SAVED                      

  CSV      → {csv_path}
  JSON     → {json_path}
  DB rows  → {db_saved}
  Accuracy → {accuracy:.1f}%  ({correct}/{len(results)} correct)

    """)


if __name__ == "__main__":
    main()