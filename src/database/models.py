"""
database/models.py — SQLAlchemy ORM models for log storage & analytics.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Enum, Index
)
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()


class LogCategory(str, enum.Enum):
    HTTP_STATUS          = "HTTP Status"
    SECURITY_ALERT       = "Security Alert"
    SYSTEM_NOTIFICATION  = "System Notification"
    ERROR                = "Error"
    RESOURCE_USAGE       = "Resource Usage"
    CRITICAL_ERROR       = "Critical Error"
    USER_ACTION          = "User Action"
    WORKFLOW_ERROR       = "Workflow Error"
    DEPRECATION_WARNING  = "Deprecation Warning"
    UNKNOWN              = "Unknown"


class SeverityLevel(str, enum.Enum):
    CRITICAL = "Critical"
    HIGH     = "High"
    MEDIUM   = "Medium"
    LOW      = "Low"
    INFO     = "Info"


class ClassificationMethod(str, enum.Enum):
    BERT  = "BERT"
    LLM   = "LLM"
    REGEX = "Regex"
    HYBRID= "Hybrid"


class LogEntry(Base):
    __tablename__ = "log_entries"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    timestamp             = Column(DateTime, default=datetime.utcnow, nullable=False)
    raw_message           = Column(Text, nullable=False)
    source                = Column(String(128), nullable=True)
    category              = Column(Enum(LogCategory), nullable=False, default=LogCategory.UNKNOWN)
    severity              = Column(Enum(SeverityLevel), nullable=False, default=SeverityLevel.INFO)
    confidence            = Column(Float, nullable=False, default=0.0)
    classification_method = Column(Enum(ClassificationMethod), nullable=False, default=ClassificationMethod.HYBRID)
    jira_ticket           = Column(String(64), nullable=True)
    processing_time_ms    = Column(Float, nullable=True)
    created_at            = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_log_entries_category",  "category"),
        Index("ix_log_entries_severity",  "severity"),
        Index("ix_log_entries_timestamp", "timestamp"),
    )

    def to_dict(self) -> dict:
        return {
            "id":                    self.id,
            "timestamp":             self.timestamp.isoformat() if self.timestamp else None,
            "raw_message":           self.raw_message,
            "source":                self.source,
            "category":              self.category.value if self.category else None,
            "severity":              self.severity.value if self.severity else None,
            "confidence":            round(self.confidence, 4),
            "classification_method": self.classification_method.value if self.classification_method else None,
            "jira_ticket":           self.jira_ticket,
            "processing_time_ms":    self.processing_time_ms,
        }