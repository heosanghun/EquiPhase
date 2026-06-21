import paramiko
import sys

def list_splits():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        print("Checking splits directory:")
        stdin, stdout, stderr = client.exec_command("ls -la /TRACKING/shoon/EquiPhase/splits/")
        print(stdout.read().decode('utf-8'))
        
        print("Checking recent files in EquiPhase root:")
        stdin, stdout, stderr = client.exec_command("ls -lt /TRACKING/shoon/EquiPhase/ | head -n 15")
        print(stdout.read().decode('utf-8'))
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    list_splits()
