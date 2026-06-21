import paramiko
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def check_pdbs():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        print("Checking remote data/pdbs directory...")
        stdin, stdout, stderr = client.exec_command("ls -la /TRACKING/shoon/EquiPhase/data/pdbs | wc -l")
        print("Number of files in data/pdbs:", stdout.read().decode('utf-8').strip())
        
        stdin, stdout, stderr = client.exec_command("ls /TRACKING/shoon/EquiPhase/data/pdbs | head -n 10")
        print("First 10 files:")
        print(stdout.read().decode('utf-8'))
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_pdbs()
