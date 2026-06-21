import paramiko
import sys

def check_proc():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        print("Running python processes:")
        stdin, stdout, stderr = client.exec_command("ps aux | grep -E 'python|docker'")
        print(stdout.read().decode('utf-8'))
        
        print("GPU Status (nvidia-smi):")
        stdin, stdout, stderr = client.exec_command("nvidia-smi")
        print(stdout.read().decode('utf-8'))
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_proc()
