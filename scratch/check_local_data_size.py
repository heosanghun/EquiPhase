import os

def check_local_size():
    root_dir = "d:/AI/EquiPhase"
    total_size = 0
    
    print("Top level directory sizes:")
    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            if item in ['.git', '__pycache__', 'runs']:
                continue
            # calculate folder size
            folder_size = 0
            file_count = 0
            for r, d, files in os.walk(path):
                for f in files:
                    fp = os.path.join(r, f)
                    if not os.path.islink(fp):
                        folder_size += os.path.getsize(fp)
                        file_count += 1
            print(f"Directory: {item} - {folder_size / 1024 / 1024:.2f} MB ({file_count} files)")
        else:
            file_size = os.path.getsize(path)
            print(f"File: {item} - {file_size / 1024:.2f} KB")

if __name__ == "__main__":
    check_local_size()
