import paramiko
import sys

def list_shoon():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        print("Connected! Listing /TRACKING/shoon...")
        
        stdin, stdout, stderr = client.exec_command("ls -la /TRACKING/shoon")
        print(stdout.read().decode('utf-8'))
        
        print("Connected! Listing /TRACKING/shoon/EquiPhase...")
        stdin, stdout, stderr = client.exec_command("ls -la /TRACKING/shoon/EquiPhase")
        print(stdout.read().decode('utf-8'))
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    list_shoon()
