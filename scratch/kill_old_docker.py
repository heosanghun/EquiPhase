import paramiko
import sys

def kill_old():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        print("Listing containers:")
        stdin, stdout, stderr = client.exec_command("docker ps")
        docker_ps_out = stdout.read().decode('utf-8')
        print(docker_ps_out)
        
        # We want to kill awesome_tu (8eb1552454f6)
        if "awesome_tu" in docker_ps_out:
            print("Killing old container awesome_tu (8eb1552454f6)...")
            stdin, stdout, stderr = client.exec_command("docker kill 8eb1552454f6")
            print(stdout.read().decode('utf-8'))
        else:
            print("awesome_bassi container not found in listing.")
            
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    kill_old()
