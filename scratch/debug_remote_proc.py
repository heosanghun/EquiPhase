import paramiko
import sys

def debug_proc():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        
        print("=== Running Docker Containers (docker ps) ===")
        stdin, stdout, stderr = client.exec_command("docker ps")
        docker_ps_output = stdout.read()
        sys.stdout.buffer.write(docker_ps_output)
        
        # Extract container ID
        docker_ps_str = docker_ps_output.decode('utf-8', errors='replace')
        container_id = None
        for line in docker_ps_str.strip().split('\n')[1:]:
            if 'pytorch' in line:
                container_id = line.split()[0]
                break
                
        if not container_id:
            lines = docker_ps_str.strip().split('\n')
            if len(lines) > 1:
                container_id = lines[1].split()[0]
                
        if container_id:
            print(f"\nInspecting Container ID: {container_id}")
            
            print("\n=== Processes inside Container ===")
            stdin, stdout, stderr = client.exec_command(f"docker exec {container_id} ps aux")
            sys.stdout.buffer.write(stdout.read())
            
            print("\n=== Splits inside Container ===")
            stdin, stdout, stderr = client.exec_command(f"docker exec {container_id} ls -la /workspace/splits")
            sys.stdout.buffer.write(stdout.read())
            
            print("\n=== Tail of honest_run.log inside Container ===")
            stdin, stdout, stderr = client.exec_command(f"docker exec {container_id} tail -n 30 /workspace/honest_run.log")
            sys.stdout.buffer.write(stdout.read())
        else:
            print("No matching docker container found.")
            
        client.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    debug_proc()
