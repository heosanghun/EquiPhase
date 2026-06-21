import paramiko
import sys

# Reconfigure stdout to use UTF-8 to prevent cp949 encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def check_tmux():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        print("Capturing tmux pane...")
        stdin, stdout, stderr = client.exec_command("tmux capture-pane -p -t equiphase_training")
        out = stdout.read().decode('utf-8', errors='replace')
        print("=" * 60)
        print("Tmux Pane Content:")
        print("=" * 60)
        print(out)
        print("=" * 60)
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_tmux()
