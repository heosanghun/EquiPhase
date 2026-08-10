import os
import re
import sys

def scan_directory(directory):
    patterns = {
        'AWS Access Key': r'AKIA[0-9A-Z]{16}',
        'GitHub Token': r'gh[pousr]_[A-Za-z0-9_]{36,255}',
        'Slack Token': r'xox[baprs]-[0-9]{12}-[0-9]{12}-[0-9a-zA-Z]{24}',
        'Stripe Key': r'sk_(test|live)_[0-9a-zA-Z]{24}',
        'Generic API Key': r'(?i)(api[_-]?key|secret|token)["\']?\s*[:=]\s*["\']?[0-9a-zA-Z]{20,}["\']?',
        'Private Key': r'-----BEGIN [A-Z ]+ PRIVATE KEY-----'
    }

    found_secrets = []
    
    for root, dirs, files in os.walk(directory):
        if '.git' in dirs:
            dirs.remove('.git')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
            
        for file in files:
            if file.endswith(('.py', '.txt', '.md', '.json', '.env', '.yaml', '.yml')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for name, pattern in patterns.items():
                            matches = re.finditer(pattern, content)
                            for match in matches:
                                # redact actual secret in log
                                found_secrets.append(f"[{name}] found in {filepath}: [REDACTED]")
                except Exception as e:
                    pass
                    
    return found_secrets

if __name__ == '__main__':
    print("WP5_SECRET_SCAN_BEGIN")
    secrets = scan_directory('.')
    if not secrets:
        print("No hardcoded secrets or credentials found. (PASS)")
    else:
        for s in secrets:
            print(s)
        print("WARNING: Secrets found. Remediation required.")
    print("WP5_SECRET_SCAN_END")
