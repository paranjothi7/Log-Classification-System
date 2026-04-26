"""
src/test_jira.py - Test JIRA connection and ticket creation
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from integrations.jira.client import JiraClient

client = JiraClient()
client._connect()

if client._jira:
    print("✅ JIRA connected successfully!")
    print(f"   Server  : {client.server}")
    print(f"   Project : {client.project}")

    # List issue types
    print("\nAvailable Issue Types:")
    print("-" * 40)
    issue_types = client._jira.issue_types_for_project("SA")
    for it in issue_types:
        print(f"  {it.name}")
    print("-" * 40)

    # Create test ticket
    key = client.create_ticket(
        summary="Test alert from SOC Log Classifier",
        description="This is a test ticket created by the log classification system.",
        category="Security Alert",
        severity="High",
    )
    if key:
        print(f"\n Test ticket created : {key}")
        print(f"   View at            : {client.server}/browse/{key}")
    else:
        print("Ticket creation failed")
else:
    print(" JIRA connection failed")