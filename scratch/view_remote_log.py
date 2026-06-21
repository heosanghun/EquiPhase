import paramiko
import sys

# Reconfigure stdout to use UTF-8 to prevent cp949 encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def view_log():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        print("\n=== honest_run.log ===")
        stdin, stdout, stderr = client.exec_command("cat /TRACKING/shoon/EquiPhase/honest_run.log")
        print(stdout.read().decode('utf-8', errors='replace'))
        
        print("\n=== docker_run.log ===")
        stdin, stdout, stderr = client.exec_command("cat /home/shoon/docker_run.log")
        print(stdout.read().decode('utf-8', errors='replace'))
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    view_log()
