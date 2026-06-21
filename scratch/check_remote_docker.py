import paramiko
import sys

def check_remote_docker():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    # Run a simple docker command to check GPU access
    cmd = "docker run --gpus all --rm pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel nvidia-smi"
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        print("Connected! Testing docker command with GPU access...")
        print(f"Running: {cmd}")
        print("=" * 60)
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')
        if out:
            print("Output:")
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
    check_remote_docker()
