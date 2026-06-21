import paramiko
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def test_gpu():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    commands = [
        "docker run --gpus '\"device=2\"' --rm pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel python -c 'import torch; print(\"CUDA:\", torch.cuda.is_available()); print(\"Device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")'",
        "docker run --gpus '\"device=2\"' --rm pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel python -c 'import torch; x = torch.randn(10).cuda(0); print(\"Allocated successfully on GPU!\")'"
    ]
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        for cmd in commands:
            print("=" * 60)
            print(f"Running: {cmd}")
            print("=" * 60)
            stdin, stdout, stderr = client.exec_command(cmd)
            out = stdout.read().decode('utf-8', errors='replace')
            err = stderr.read().decode('utf-8', errors='replace')
            if out:
                print("STDOUT:")
                print(out)
            if err:
                print("STDERR:")
                print(err)
            print()
            
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_gpu()
