import paramiko
import sys
import shutil
import os

def download_results():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    remote_files = [
        "/TRACKING/shoon/EquiPhase/bifurcation_audit_plot.png",
        "/TRACKING/shoon/EquiPhase/bifurcation_audit_plot.pdf",
        "/TRACKING/shoon/EquiPhase/honest_audit_results.csv",
        "/TRACKING/shoon/EquiPhase/placebo_retraining.log",
        "/TRACKING/shoon/EquiPhase/honest_audit_report.log",
        "/TRACKING/shoon/EquiPhase/completion_summary.log"
    ]
    
    local_dir = 'd:/AI/EquiPhase'
    artifact_dir = 'C:/Users/Sims/.gemini/antigravity/brain/e20d7f14-205f-4a52-9696-5f6f1c4caac8'
    
    print(f"Connecting to {hostname}:{port}...")
    transport = paramiko.Transport((hostname, port))
    try:
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        print("SFTP Connection established!")
        
        for rf in remote_files:
            filename = os.path.basename(rf)
            local_path = os.path.join(local_dir, filename)
            artifact_path = os.path.join(artifact_dir, filename)
            
            print(f"Downloading {rf} -> {local_path}...")
            sftp.get(rf, local_path)
            
            # Copy to artifact dir as well
            print(f"Copying to artifact directory -> {artifact_path}...")
            shutil.copy2(local_path, artifact_path)
            
        print("\nAll results downloaded successfully!")
        sftp.close()
        transport.close()
    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_results()
