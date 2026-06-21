import paramiko
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def test_esm():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Run ESM loading and embedding extraction test
    cmd = (
        "docker run --gpus '\"device=2\"' --rm "
        "-v /TRACKING/shoon/EquiPhase:/workspace "
        "-v /home/shoon/.cache/huggingface:/root/.cache/huggingface "
        "-w /workspace "
        "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel "
        "bash -c \"pip install -q transformers && python test_esm_in_docker.py\""
    )
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        print("Connected! Running ESM test...")
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
        print("=" * 60)
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_esm()
