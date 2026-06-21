import paramiko
import sys

def check_remote():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    commands = {
        "Nvidia SMI": "nvidia-smi",
        "Disk Space": "df -h",
        "Groups / Permissions": "groups; id",
        "Docker Version": "docker --version",
        "Docker Containers": "docker ps -a",
        "Docker Images": "docker images",
        "Tmux / Screen Presence": "which tmux; which screen",
        "Tracking/NFS directories Check": "ls -ld /TRACKING /mnt/nfs /home/shoon"
    }
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        print("Connected! Running diagnostics...\n")
        
        for desc, cmd in commands.items():
            print("=" * 60)
            print(f"Running: {desc} ({cmd})")
            print("=" * 60)
            stdin, stdout, stderr = client.exec_command(cmd)
            out = stdout.read().decode('utf-8')
            err = stderr.read().decode('utf-8')
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
    check_remote()
