import paramiko
import sys

# Reconfigure stdout to use UTF-8 to prevent cp949 encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def find_logs():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        print("Checking where docker_run.log is:")
        stdin, stdout, stderr = client.exec_command("find /home/shoon -maxdepth 2 -name 'docker_run.log'")
        print("Found under /home/shoon:")
        print(stdout.read().decode('utf-8'))
        
        print("Running docker ps -a:")
        stdin, stdout, stderr = client.exec_command("docker ps -a")
        print(stdout.read().decode('utf-8'))
        
        # Read the file
        stdin, stdout, stderr = client.exec_command("cat /home/shoon/docker_run.log")
        content = stdout.read().decode('utf-8', errors='replace')
        print("=" * 60)
        print("Content of /home/shoon/docker_run.log:")
        print("=" * 60)
        print(content)
        print("=" * 60)
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    find_logs()
