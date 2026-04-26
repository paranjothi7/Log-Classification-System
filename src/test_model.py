
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from processors.enhanced_processor import EnhancedProcessor

proc = EnhancedProcessor()

test_logs = [
    ('nova.osapi_compute.wsgi.server status: 200 len: 1893 time: 0.26',  'HTTP Status'),
    ('Multiple bad login attempts detected on user 8538 account',          'Security Alert'),
    ('Critical system unit error: unit ID Component55',                    'Critical Error'),
    ('nova.compute.claims Total memory: 64172 MB, used: 512.00 MB',       'Resource Usage'),
    ('File data_6169.csv uploaded successfully by user User953.',          'System Notification'),
    ('User User685 logged out.',                                           'User Action'),
    ('Shard 6 replication task ended in failure',                          'Error'),
    ('Escalation workflow failed for ticket TKT-4821',                     'Workflow Error'),
]

print('-' * 85)
print(f"{'Log Message':<45} {'Expected':<22} {'Got':<22} {'OK'}")
print('-' * 85)

correct = 0
for msg, expected in test_logs:
    result = proc.process(msg)
    ok     = 'OK' if result.category == expected else 'FAIL'
    if result.category == expected:
        correct += 1
    print(f'{msg[:44]:<45} {expected:<22} {result.category:<22} {ok}')

print('-' * 85)
print(f'Accuracy : {correct}/{len(test_logs)} ({correct/len(test_logs)*100:.0f}%)')
print('-' * 85)
"@ | Out-File -FilePath "