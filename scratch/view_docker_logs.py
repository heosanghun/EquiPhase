import paramiko
import sys

# Reconfigure stdout to use UTF-8 to prevent cp949 encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def view_docker_logs():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        # Find active container ID
        print("Finding running container ID...")
        stdin, stdout, stderr = client.exec_command(
            "docker ps --filter ancestor=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel --format '{{.ID}} {{.Names}}'"
        )
        containers = stdout.read().decode('utf-8').strip().split('\n')
        
        if not containers or containers[0] == '':
            print("No active training containers found.")
        else:
            print(f"Active containers: {containers}")
            for container in containers:
                cid, name = container.split(' ', 1)
                print(f"\n=== Container: {name} ({cid}) Logs ===")
                stdin, stdout, stderr = client.exec_command(f"docker logs --tail 30 {cid}")
                print(stdout.read().decode('utf-8', errors='replace'))
                print(stderr.read().decode('utf-8', errors='replace'))
                
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    view_docker_logs()
