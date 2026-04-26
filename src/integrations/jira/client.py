"""
integrations/jira/client.py — Auto-create JIRA tickets for critical alerts.
"""
from __future__ import annotations
import os
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class JiraClient:
    def __init__(self):
        self.server    = os.getenv("JIRA_SERVER", "")
        self.email     = os.getenv("JIRA_EMAIL", "")
        self.token     = os.getenv("JIRA_API_TOKEN", "")
        self.project   = os.getenv("JIRA_PROJECT_KEY", "SOC")
        self._jira     = None

    def _connect(self):
        if self._jira is not None:
            return
        try:
            from jira import JIRA
            self._jira = JIRA(
                server=self.server,
                basic_auth=(self.email, self.token),
            )
            logger.info("JIRA connected.")
        except Exception as exc:
            logger.warning(f"JIRA connect failed: {exc}")
            self._jira = False

    # severity → JIRA priority mapping
    _PRIORITY_MAP = {
        "Critical": "Highest",
        "High":     "High",
        "Medium":   "Medium",
        "Low":      "Low",
        "Info":     "Lowest",
    }

    def create_ticket(
            self,
            summary: str,
            description: str,
            category: str,
            severity: str,
    ) -> Optional[str]:
        self._connect()
        if not self._jira:
            return None
        try:
            issue = self._jira.create_issue(
                project=self.project,
                summary=f"[{category}] {summary[:80]}",
                description=description,
                issuetype={"name": "Task"},  # ← changed from "Bug" to "Task"
                priority={"name": self._PRIORITY_MAP.get(severity, "Medium")},
                labels=["soc-auto", category.lower().replace(" ", "-")],
            )
            logger.info(f"JIRA ticket created: {issue.key}")
            return issue.key
        except Exception as exc:
            logger.error(f"JIRA create_ticket failed: {exc}")
            return None