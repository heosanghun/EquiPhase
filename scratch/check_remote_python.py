import paramiko
import sys

def check_remote_python():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    commands = {
        "Python 3 Version": "python3 --version",
        "Which Python": "which python; which python3; which pip; which pip3",
        "Host PyTorch check": "python3 -c 'import torch; print(\"PyTorch:\", torch.__version__, \"CUDA available:\", torch.cuda.is_available())' 2>&1",
        "Host Pip List": "pip list | grep -E 'torch|deq|paramiko' 2>&1"
    }
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        print("Connected! Running checks...\n")
        
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
    check_remote_python()
