import paramiko
import sys

def check_hf_cache():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        print("Connected! Listing huggingface cache...")
        
        stdin, stdout, stderr = client.exec_command("ls -la ~/.cache/huggingface/hub")
        print("Stdout:")
        print(stdout.read().decode('utf-8'))
        print("Stderr:")
        print(stderr.read().decode('utf-8'))
        
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_hf_cache()
