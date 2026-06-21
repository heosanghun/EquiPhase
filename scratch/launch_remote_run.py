import paramiko
import sys
import time

def launch_remote():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    print(f"Connecting to {hostname}:{port}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, port=port, username=username, password=password, timeout=10)
        print("Connected!")
        
        # 1. Make the run script executable
        print("Making remote_run.sh executable...")
        client.exec_command("chmod +x /TRACKING/shoon/EquiPhase/remote_run.sh")
        time.sleep(1)
        
        # 2. Reset training completed flag if exists
        print("Removing stale training_completed.txt and completion_summary.log...")
        client.exec_command("rm -f /TRACKING/shoon/EquiPhase/training_completed.txt /TRACKING/shoon/EquiPhase/completion_summary.log")
        time.sleep(1)
        
        # 3. Terminate existing tmux session if exists
        print("Stopping existing tmux session 'equiphase_training' if running...")
        client.exec_command("tmux kill-session -t equiphase_training 2>/dev/null")
        time.sleep(2)
        
        # 4. Start new tmux session
        print("Starting new tmux session 'equiphase_training'...")
        client.exec_command("tmux new-session -d -s equiphase_training")
        time.sleep(2)
        
        # 5. Send command to run Docker inside tmux
        docker_cmd = (
            "docker run --gpus '\"device=5\"' --rm "
            "-v /TRACKING/shoon/EquiPhase:/workspace "
            "-v /home/shoon/.cache/huggingface:/root/.cache/huggingface "
            "-w /workspace "
            "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel "
            "bash remote_run.sh"
        )
        
        print(f"Sending docker run command to tmux...")
        tmux_cmd = f"tmux send-keys -t equiphase_training \"{docker_cmd} 2>&1 | tee docker_run.log\" C-m"
        client.exec_command(tmux_cmd)
        time.sleep(2)
        
        # 6. Verify it is running
        print("Verifying running tmux sessions...")
        stdin, stdout, stderr = client.exec_command("tmux list-sessions")
        print("Tmux Sessions:")
        print(stdout.read().decode('utf-8'))
        
        print("Checking remote processes inside tmux...")
        stdin, stdout, stderr = client.exec_command("ps aux | grep -E 'docker run|remote_run'")
        print(stdout.read().decode('utf-8'))
        
        client.close()
        print("Launch sequence complete! Remote session is running in the background.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    launch_remote()
