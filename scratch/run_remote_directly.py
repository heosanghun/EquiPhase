import paramiko
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_directly():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    cmd = (
        "docker run --gpus '\"device=2\"' --rm "
        "-v /TRACKING/shoon/EquiPhase:/workspace "
        "-v /home/shoon/.cache/huggingface:/root/.cache/huggingface "
        "-w /workspace "
        "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel "
        "bash -c \"pip install -q torchdeq scipy transformers scikit-learn matplotlib pandas openpyxl && python run_real_masterpiece.py\""
    )
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        print("Connected! Starting run...")
        print(f"Executing: {cmd}")
        print("=" * 80)
        
        stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
        
        # Read from stdout and print in real-time
        while True:
            line = stdout.readline()
            if not line:
                break
            print(line, end="")
            
        print("=" * 80)
        print("Run finished.")
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_directly()
