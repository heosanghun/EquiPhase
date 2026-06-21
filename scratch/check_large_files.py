import os

def check_large_files():
    root_dir = "d:/AI/EquiPhase"
    large_files = []
    
    for r, d, files in os.walk(root_dir):
        if '.git' in r or '__pycache__' in r or 'runs' in r:
            continue
        for f in files:
            fp = os.path.join(r, f)
            if not os.path.islink(fp):
                sz = os.path.getsize(fp)
                if sz > 10 * 1024 * 1024:  # > 10 MB
                    large_files.append((fp, sz))
                    
    large_files.sort(key=lambda x: x[1], reverse=True)
    print("Files larger than 10MB:")
    for fp, sz in large_files:
        print(f"{fp} - {sz / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    check_large_files()
