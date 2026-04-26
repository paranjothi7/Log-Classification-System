"""
src/utils/result_saver.py
Save classification results to CSV, JSON, and PostgreSQL.
"""
import os, json
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

# Results folder sits next to src/
BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class ResultSaver:

    @staticmethod
    def to_csv(results: list, filename: str = None) -> str:
        """Save classification results to a timestamped CSV file."""
        if not filename:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{ts}.csv"
        path = RESULTS_DIR / filename
        pd.DataFrame(results).to_csv(path, index=False)
        logger.info(f"CSV saved -> {path}")
        return str(path)

    @staticmethod
    def to_json(results: list, filename: str = None) -> str:
        """Save classification results to a JSON file."""
        if not filename:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{ts}.json"
        path = RESULTS_DIR / filename
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"JSON saved -> {path}")
        return str(path)

    @staticmethod
    def to_database(results: list) -> int:
        """Bulk-save results to database via LogService."""
        from database.service import LogService
        saved = LogService.bulk_save(results)
        logger.info(f"DB saved -> {saved} rows")
        return saved

    @staticmethod
    def summary_report(results: list) -> dict:
        """Print and return a summary of classification results."""
        df = pd.DataFrame(results)

        summary = {
            "total":            len(df),
            "by_category":      df["category"].value_counts().to_dict(),
            "by_severity":      df["severity"].value_counts().to_dict(),
            "by_method":        df["classification_method"].value_counts().to_dict(),
            "avg_confidence":   round(float(df["confidence"].mean()), 4),
            "avg_time_ms":      round(float(df["processing_time_ms"].mean()), 2),
        }

        # Print readable summary
        print("\n" + "=" * 55)
        print("  CLASSIFICATION SUMMARY")
        print("=" * 55)
        print(f"  Total logs processed : {summary['total']}")
        print(f"  Avg confidence       : {summary['avg_confidence']:.1%}")
        print(f"  Avg processing time  : {summary['avg_time_ms']} ms")

        print("\n  By Category:")
        for cat, count in sorted(summary["by_category"].items(),
                                  key=lambda x: x[1], reverse=True):
            bar = "#" * int(count / summary["total"] * 30)
            print(f"    {cat:<25} {count:>4}  {bar}")

        print("\n  By Severity:")
        for sev, count in sorted(summary["by_severity"].items(),
                                  key=lambda x: x[1], reverse=True):
            print(f"    {sev:<15} {count:>4}")

        print("\n  By Method:")
        for method, count in sorted(summary["by_method"].items(),
                                     key=lambda x: x[1], reverse=True):
            print(f"    {method:<15} {count:>4}")

        print("=" * 55 + "\n")

        logger.info(f"Summary: {json.dumps(summary, indent=2)}")
        return summary