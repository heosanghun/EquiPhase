import paramiko
import sys

def main():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    cmd = (
        "docker run --gpus '\"device=5\"' --rm "
        "-v /TRACKING/shoon/EquiPhase:/workspace "
        "-v /home/shoon/.cache/huggingface:/root/.cache/huggingface "
        "-w /workspace "
        "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel "
        "bash -c \"pip install -q torchdeq && python verify_mathematics.py\""
    )
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        print("Connected to remote server. Executing mathematical verification...")
        print(f"Command: {cmd}")
        print("=" * 80)
        
        stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
        
        while True:
            line = stdout.readline()
            if not line:
                break
            print(line, end="")
            
        print("=" * 80)
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
