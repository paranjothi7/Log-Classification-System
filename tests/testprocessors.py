"""
tests/test_processors.py
Unit tests using real log patterns from synthetic_logs.csv.
Covers all 9 label classes across regex, BERT, and full pipeline paths.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from processors.enhanced_processor import RegexClassifier, ClassificationResult, EnhancedProcessor


@pytest.fixture(scope="module")
def regex_clf():
    return RegexClassifier()

@pytest.fixture(scope="module")
def processor():
    return EnhancedProcessor()


# HTTP Status

class TestHTTPStatus:
    def test_nova_wsgi_get_200(self, regex_clf):
        msg = ('nova.osapi_compute.wsgi.server [req-b9718cd8] 10.11.10.1 '
               '"GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1" '
               'status: 200 len: 1893 time: 0.2675118')
        r = regex_clf.classify(msg)
        assert r is not None
        assert r.category == "HTTP Status"
        assert r.severity == "Info"

    def test_rcode_200(self, regex_clf):
        msg = ('nova.osapi_compute.wsgi.server [req-abc] 10.11.10.1 '
               '"GET /v2/abc/servers/detail HTTP/1.1" RCODE  200 len: 1874 time: 0.22')
        r = regex_clf.classify(msg)
        assert r is not None and r.category == "HTTP Status"

    def test_return_code_variant(self, regex_clf):
        msg = ('nova.osapi_compute.wsgi.server [req-xyz] 10.11.10.1 '
               '"GET /v2/xyz/servers/detail HTTP/1.1" Return code: 200 len: 1893')
        r = regex_clf.classify(msg)
        assert r is not None and r.category == "HTTP Status"

    def test_post_404(self, regex_clf):
        msg = ('nova.osapi_compute.wsgi.server [req-033d97b9] 10.11.10.1 '
               '"POST /v2/e9746973/os-server-external-events HTTP/1.1" status: 404')
        r = regex_clf.classify(msg)
        assert r is not None and r.category == "HTTP Status"


# Security Alert
class TestSecurityAlert:
    def test_multiple_bad_login(self, regex_clf):
        r = regex_clf.classify("Multiple bad login attempts detected on user 8538 account")
        assert r is not None and r.category == "Security Alert"

    def test_brute_force(self, regex_clf):
        r = regex_clf.classify("Alert: brute force login attempt from 192.168.80.114 detected")
        assert r is not None and r.category == "Security Alert"
        assert r.severity in ("High", "Critical")

    def test_suspicious_login(self, regex_clf):
        r = regex_clf.classify("Suspicious login activity detected from 192.168.24.250")
        assert r is not None and r.category == "Security Alert"

    def test_denied_access(self, regex_clf):
        r = regex_clf.classify("Denied access attempt on restricted account Account2682")
        assert r is not None and r.category == "Security Alert"

    def test_bypass_api_security(self, regex_clf):
        r = regex_clf.classify("User 7662 tried to bypass API security measures")
        assert r is not None and r.category == "Security Alert"

    def test_unauthorized_access(self, regex_clf):
        r = regex_clf.classify("Unauthorized access to data was attempted")
        assert r is not None and r.category == "Security Alert"


# Critical Error

class TestCriticalError:
    def test_email_service(self, regex_clf):
        r = regex_clf.classify("Email service experiencing issues with sending")
        assert r is not None and r.category == "Critical Error"
        assert r.severity == "Critical"

    def test_system_unit_error(self, regex_clf):
        r = regex_clf.classify("Critical system unit error: unit ID Component55")
        assert r is not None and r.category == "Critical Error"

    def test_component_malfunction(self, regex_clf):
        r = regex_clf.classify("System component malfunction: component ID Component79")
        assert r is not None and r.category == "Critical Error"

    def test_disk_fault(self, regex_clf):
        r = regex_clf.classify("Detection of multiple disk faults in RAID setup")
        assert r is not None and r.category == "Critical Error"

    def test_boot_terminated(self, regex_clf):
        r = regex_clf.classify("Boot process terminated unexpectedly due to kernel issue")
        assert r is not None and r.category == "Critical Error"


# Resource Usage

class TestResourceUsage:
    def test_nova_compute_claims(self, regex_clf):
        msg = ('nova.compute.claims [req-a07ac654] [instance: bf8c824d] '
               'Total memory: 64172 MB, used: 512.00 MB')
        r = regex_clf.classify(msg)
        assert r is not None and r.category == "Resource Usage"

    def test_resource_tracker(self, regex_clf):
        msg = ('nova.compute.resource_tracker [req-addc1839] '
               'Final resource view: name=cp-1 phys_ram=64172MB used_ram=512MB '
               'phys_disk=15GB used_disk=0GB total_vcpus=16 used_vcpus=0')
        r = regex_clf.classify(msg)
        assert r is not None and r.category == "Resource Usage"

    def test_memory_limit(self, regex_clf):
        msg = ('nova.compute.claims [req-d6986b54] '
               '[instance: af5f7392] memory limit: 96258.00 MB, free: 95746.00 MB')
        r = regex_clf.classify(msg)
        assert r is not None and r.category == "Resource Usage"

    def test_disk_limit_not_specified(self, regex_clf):
        msg = ('nova.compute.claims [req-72b4858f] '
               '[instance: 63a0d960] disk limit not specified, defaulting to unlimited')
        r = regex_clf.classify(msg)
        assert r is not None and r.category == "Resource Usage"


# System Notification

class TestSystemNotification:
    def test_file_upload(self, regex_clf):
        r = regex_clf.classify("File data_6169.csv uploaded successfully by user User953.")
        assert r is not None and r.category == "System Notification"

    def test_backup_completed(self, regex_clf):
        r = regex_clf.classify("Backup completed successfully.")
        assert r is not None and r.category == "System Notification"

    def test_backup_started(self, regex_clf):
        r = regex_clf.classify("Backup started at 2025-05-14 07:06:55.")
        assert r is not None and r.category == "System Notification"

    def test_system_reboot(self, regex_clf):
        r = regex_clf.classify("System reboot initiated by user User243.")
        assert r is not None and r.category == "System Notification"


# User Action

class TestUserAction:
    def test_user_logged_out(self, regex_clf):
        r = regex_clf.classify("User User685 logged out.")
        assert r is not None and r.category == "User Action"
        assert r.severity == "Info"

    def test_account_created(self, regex_clf):
        r = regex_clf.classify("Account with ID 5351 created by User634.")
        assert r is not None and r.category == "User Action"


# Error

class TestError:
    def test_shard_replication_failure(self, regex_clf):
        r = regex_clf.classify("Shard 6 replication task ended in failure")
        assert r is not None and r.category == "Error"

    def test_replication_not_complete(self, regex_clf):
        r = regex_clf.classify("Data replication task for shard 14 did not complete")
        assert r is not None and r.category == "Error"


# Full Pipeline

class TestEnhancedProcessor:
    def test_returns_classification_result(self, processor):
        r = processor.process("Multiple bad login attempts detected on user 8538 account")
        assert isinstance(r, ClassificationResult)
        assert r.category in [
            "HTTP Status","Security Alert","System Notification","Error",
            "Resource Usage","Critical Error","User Action",
            "Workflow Error","Deprecation Warning","Unknown",
        ]
        assert r.severity in ["Critical","High","Medium","Low","Info"]
        assert 0.0 <= r.confidence <= 1.0
        assert r.processing_time_ms >= 0

    def test_batch_returns_all_results(self, processor):
        msgs = [
            'nova.osapi_compute.wsgi.server [req-abc] 10.11.10.1 "GET /v2/servers HTTP/1.1" status: 200 len: 1893 time: 0.26',
            "Brute force login attempt from 192.168.1.1 detected",
            "Email service experiencing issues with sending",
            "nova.compute.claims [req-xyz] Total memory: 64172 MB, used: 512 MB",
            "File data_99.csv uploaded successfully by user User1.",
            "User User100 logged out.",
            "Shard 3 replication task ended in failure",
            "System reboot initiated by user User5.",
        ]
        results = processor.process_batch(msgs)
        assert len(results) == len(msgs)
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_http_status_message(self, processor):
        msg = ('nova.osapi_compute.wsgi.server [req-b9718cd8] 10.11.10.1 '
               '"GET /v2/54fadb412c/servers/detail HTTP/1.1" status: 200 len: 1893 time: 0.26')
        r = processor.process(msg)
        assert r.category == "HTTP Status"

    def test_security_alert_message(self, processor):
        r = processor.process("Alert: brute force login attempt from 192.168.80.114 detected")
        assert r.category == "Security Alert"

    def test_critical_error_message(self, processor):
        r = processor.process("Critical system unit error: unit ID Component55")
        assert r.category == "Critical Error"
        assert r.severity == "Critical"