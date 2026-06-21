import paramiko
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def diagnose():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    commands = {
        "Disk Space (df -h)": "df -h",
        "System Memory (free -m)": "free -m",
        "Recent dmesg (OOM check)": "dmesg -T | grep -i -E 'oom|kill|segfault' | tail -n 30",
        "Docker info": "docker info | grep -E 'Storage|Logging|Kernel'",
        "All Docker Containers (docker ps -a)": "docker ps -a",
        "Tmux Pane capture": "tmux capture-pane -p -t equiphase_training"
    }
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        print("Connected! Running diagnostics...\n")
        
        for desc, cmd in commands.items():
            print("=" * 60)
            print(f"{desc} ({cmd})")
            print("=" * 60)
            stdin, stdout, stderr = client.exec_command(cmd)
            out = stdout.read().decode('utf-8', errors='replace')
            err = stderr.read().decode('utf-8', errors='replace')
            if out:
                print(out)
            if err:
                print("Error Output:")
                print(err)
            print()
            
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    diagnose()
