"""
processors/enhanced_processor.py
Hybrid Regex → BERT → LLM classification pipeline.
Labels derived from synthetic_logs.csv:
  HTTP Status | Security Alert | System Notification | Error |
  Resource Usage | Critical Error | User Action | Workflow Error |
  Deprecation Warning | Unknown
"""
from __future__ import annotations
import re, os, time, json
from dataclasses import dataclass, field
from typing import Optional, Tuple
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

CATEGORIES = [
    "HTTP Status", "Security Alert", "System Notification", "Error",
    "Resource Usage", "Critical Error", "User Action",
    "Workflow Error", "Deprecation Warning", "Unknown",
]
SEVERITIES = ["Critical", "High", "Medium", "Low", "Info"]

_REGEX_RULES: list[Tuple[re.Pattern, str, str]] = [
    (re.compile(
        r'(nova\.(osapi_compute|metadata|compute)\.wsgi\.server|'
        r'"(GET|POST|PUT|DELETE|PATCH)\s+/|'
        r'status:\s*\d{3}|RCODE\s+\d{3}|Return code:\s*\d{3}|'
        r'HTTP status code\s+-\s+\d{3}|Status code\s+-\s+\d{3})', re.I),
     "HTTP Status", "Info"),

    (re.compile(
        r'(brute\s*force|multiple\s+(bad\s+)?login\s+(attempt|fail)|'
        r'unauthori[zs]ed\s+access|suspicious\s+login|'
        r'denied\s+access\s+attempt|bypass\s+API\s+security|'
        r'sql\s+injection|xss|intrusion\s+detect|'
        r'abnormal\s+system\s+behavior.*security|potential\s+security\s+breach)', re.I),
     "Security Alert", "High"),

    (re.compile(
        r'(critical\s+system\s+(unit\s+)?error|'
        r'system\s+component\s+malfunction|'
        r'detection\s+of\s+multiple\s+disk\s+fault|'
        r'boot\s+process\s+terminated\s+unexpectedly|'
        r'kernel\s+issue|email\s+service\s+experiencing\s+issues)', re.I),
     "Critical Error", "Critical"),

    (re.compile(
        r'(nova\.compute\.(claims|resource_tracker)|'
        r'total\s+memory:|used\s+ram=|phys_ram=|memory\s+limit:|'
        r'free:\s*\d+\s*MB|disk\s+limit\s+not\s+specified|'
        r'final\s+resource\s+view|memory\s+(exceeded|leak)|'
        r'cpu\s+(high|exceeded|spike)|out\s+of\s+memory|oom)', re.I),
     "Resource Usage", "Medium"),

    (re.compile(
        r'(file\s+\S+\.(csv|txt|log|json)\s+uploaded\s+successfully|'
        r'backup\s+(completed\s+successfully|started\s+at)|'
        r'system\s+reboot\s+initiated)', re.I),
     "System Notification", "Info"),

    (re.compile(
        r'(user\s+\S+\s+logged\s+(in|out)|'
        r'account\s+with\s+ID\s+\d+\s+created\s+by|logged\s+out\.)', re.I),
     "User Action", "Info"),

    (re.compile(
        r'(escalation\s+(workflow\s+)?fail|task\s+assign.*fail|'
        r'workflow\s+(error|broken|stuck)|process\s+(fail|timeout))', re.I),
     "Workflow Error", "High"),

    (re.compile(
        r'(replication\s+task.*(ended\s+in\s+failure|did\s+not\s+complete)|'
        r'shard\s+\d+\s+replication.*fail|connection\s+(timeout|refused|reset)|'
        r'service\s+unavailable|503|504)', re.I),
     "Error", "Medium"),

    (re.compile(
        r'(deprecat(ed|ion)|will\s+be\s+removed\s+in|use\s+.+\s+instead|end.of.life)',
        re.I),
     "Deprecation Warning", "Low"),
]

@dataclass
class ClassificationResult:
    category:           str   = "Unknown"
    severity:           str   = "Info"
    confidence:         float = 0.0
    method:             str   = "Regex"
    reasoning:          str   = ""
    processing_time_ms: float = 0.0
    raw_scores:         dict  = field(default_factory=dict)

class RegexClassifier:
    def classify(self, message: str) -> Optional[ClassificationResult]:
        for pattern, category, severity in _REGEX_RULES:
            if pattern.search(message):
                return ClassificationResult(
                    category=category, severity=severity,
                    confidence=0.95, method="Regex",
                    reasoning=f"Matched regex rule for '{category}'",
                )
        return None

class BERTClassifier:
    _pipeline   = None
    _model_path = os.getenv("BERT_MODEL_PATH", "models/bert_log_classifier")
    _threshold  = float(os.getenv("BERT_CONFIDENCE_THRESHOLD", "0.75"))

    @classmethod
    def _load(cls):
        if cls._pipeline is None:
            try:
                from transformers import pipeline
                cls._pipeline = pipeline("text-classification",
                                         model=cls._model_path, top_k=None)
                logger.info("BERT pipeline loaded.")
            except Exception as exc:
                logger.warning(f"BERT load failed ({exc}); BERT disabled.")
                cls._pipeline = False

    def classify(self, message: str) -> Optional[ClassificationResult]:
        self._load()
        if not self._pipeline:
            return None
        try:
            scores = self._pipeline(message[:512])[0]
            best   = max(scores, key=lambda x: x["score"])
            if best["score"] < self._threshold:
                return None
            return ClassificationResult(
                category=best["label"],
                severity=_severity_for(best["label"], best["score"]),
                confidence=best["score"],
                method="BERT",
                raw_scores={s["label"]: round(s["score"], 4) for s in scores},
            )
        except Exception as exc:
            logger.error(f"BERT classify error: {exc}")
            return None

_CATEGORY_LIST = ", ".join(f'"{c}"' for c in CATEGORIES if c != "Unknown")
_LLM_SYSTEM = f"""You are a SOC log classifier. Classify each log into exactly one category.
Valid categories: {_CATEGORY_LIST}
Respond ONLY with valid JSON (no markdown):
{{"category":"<category>","severity":"<Critical|High|Medium|Low|Info>","confidence":<0-1>,"reasoning":"<one sentence>"}}"""


class LLMClassifier:
    _client = None

    @classmethod
    def _load(cls):
        if cls._client is None:
            try:
                from groq import Groq
                cls._client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
                logger.info("GROQ client ready.")
            except Exception as exc:
                logger.warning(f"GROQ init failed: {exc}")
                cls._client = False

    def classify(self, message: str) -> Optional[ClassificationResult]:
        self._load()
        if not self._client:
            return None
        try:
            resp = self._client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user",   "content": f"Log: {message[:800]}"},
                ],
                temperature=0.1, max_tokens=200,
            )
            data = json.loads(resp.choices[0].message.content.strip())
            return ClassificationResult(
                category=data.get("category", "Unknown"),
                severity=data.get("severity", "Info"),
                confidence=float(data.get("confidence", 0.6)),
                method="LLM", reasoning=data.get("reasoning", ""),
            )
        except Exception as exc:
            logger.error(f"LLM classify error: {exc}")
            return None


def _severity_for(category: str, confidence: float) -> str:
    return {
        "Critical Error":      "Critical",
        "Security Alert":      "Critical" if confidence > 0.92 else "High",
        "Workflow Error":      "High",
        "Error":               "Medium",
        "Resource Usage":      "Medium",
        "HTTP Status":         "Info",
        "System Notification": "Info",
        "User Action":         "Info",
        "Deprecation Warning": "Low",
    }.get(category, "Info")


class EnhancedProcessor:
    def __init__(self):
        self.regex = RegexClassifier()
        self.bert  = BERTClassifier()
        self.llm   = LLMClassifier()

    def process(self, message: str, source: str = "unknown") -> ClassificationResult:
        t0 = time.perf_counter()
        result = (
            self.regex.classify(message)
            or self.bert.classify(message)
            or self.llm.classify(message)
            or ClassificationResult(
                category="Unknown", severity="Info",
                confidence=0.0, method="Hybrid",
                reasoning="No classifier produced a result.",
            )
        )
        result.processing_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.debug(
            f"[{result.method}] {result.category} | {result.severity} "
            f"| conf={result.confidence:.2f} | {result.processing_time_ms}ms"
        )
        return result

    def process_batch(self, messages: list[str]) -> list[ClassificationResult]:
        return [self.process(m) for m in messages]