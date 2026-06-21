import paramiko
import sys
import os

# Reconfigure stdout to use UTF-8 to prevent cp949 encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def check_status():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        # Check if completed flag exists
        stdin, stdout, stderr = client.exec_command("ls -la /TRACKING/shoon/EquiPhase/training_completed.txt")
        exists = "training_completed.txt" in stdout.read().decode('utf-8')
        
        if exists:
            print("============================================================")
            print("STATUS: TRAINING AND AUDITING COMPLETED!")
            print("============================================================")
            
            # Print completion summary
            stdin, stdout, stderr = client.exec_command("cat /TRACKING/shoon/EquiPhase/completion_summary.log")
            print(stdout.read().decode('utf-8', errors='replace'))
        else:
            print("============================================================")
            print("STATUS: RUNNING IN BACKGROUND")
            print("============================================================")
            
            # Print last few lines of logs to show progress
            print("\nTail of docker_run.log:")
            stdin, stdout, stderr = client.exec_command("tail -n 15 /TRACKING/shoon/EquiPhase/docker_run.log")
            print(stdout.read().decode('utf-8', errors='replace'))
            
            print("\nTail of honest_run.log:")
            stdin, stdout, stderr = client.exec_command("tail -n 15 /TRACKING/shoon/EquiPhase/honest_run.log")
            print(stdout.read().decode('utf-8', errors='replace'))
            
        client.close()
    except Exception as e:
        print(f"Error checking status: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_status()
