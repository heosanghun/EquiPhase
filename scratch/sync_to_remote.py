import paramiko
import os
import sys

def sync_to_remote():
    hostname = '203.255.93.75'
    port = 10022
    username = 'shoon'
    password = 'rvlab@guest_sh'
    
    local_dir = 'd:/AI/EquiPhase'
    remote_dir = '/TRACKING/shoon/EquiPhase'
    
    print(f"Connecting to {hostname}:{port}...")
    transport = paramiko.Transport((hostname, port))
    try:
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        print("SFTP Connection established!")
        
        # Ensure remote root directory exists
        try:
            sftp.mkdir(remote_dir)
            print(f"Created remote root directory: {remote_dir}")
        except IOError:
            # Already exists
            pass
            
        # Helper function to create remote directory recursively
        def make_remote_dir(path):
            parts = path.split('/')
            current = ""
            for part in parts:
                if not part:
                    continue
                if current:
                    current = current + '/' + part
                else:
                    if path.startswith('/'):
                        current = '/' + part
                    else:
                        current = part
                try:
                    sftp.mkdir(current)
                except IOError:
                    pass # Already exists or parent doesn't exist yet
                    
        # Recursively walk local files
        file_count = 0
        byte_count = 0
        
        for root, dirs, files in os.walk(local_dir):
            # Exclude directories
            if '.git' in root or '__pycache__' in root or 'runs' in root:
                continue
                
            # Compute relative path
            rel_path = os.path.relpath(root, local_dir).replace('\\', '/')
            if rel_path == '.':
                current_remote_dir = remote_dir
            else:
                current_remote_dir = f"{remote_dir}/{rel_path}"
                make_remote_dir(current_remote_dir)
                
            for file in files:
                local_file_path = os.path.join(root, file)
                remote_file_path = f"{current_remote_dir}/{file}"
                
                # Exclude large files specifically
                # 1. esm2_t33_650M_UR50D/model.safetensors (2.5GB)
                # 2. esm2_residue_embeddings.pkl (2.1GB)
                if 'model.safetensors' in file or 'esm2_residue_embeddings.pkl' in file:
                    print(f"Skipping large file: {local_file_path}")
                    continue
                    
                # Upload file
                try:
                    sz = os.path.getsize(local_file_path)
                    print(f"Uploading {local_file_path} ({sz / 1024:.2f} KB) -> {remote_file_path}...")
                    sftp.put(local_file_path, remote_file_path)
                    file_count += 1
                    byte_count += sz
                except Exception as e:
                    print(f"Failed to upload {local_file_path}: {e}")
                    
        print("\n" + "=" * 60)
        print(f"Sync complete! Uploaded {file_count} files, total size: {byte_count / 1024 / 1024:.2f} MB")
        print("=" * 60)
        sftp.close()
        transport.close()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sync_to_remote()
