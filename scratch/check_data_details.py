import os

def check_data():
    data_dir = "d:/AI/EquiPhase/data"
    for item in os.listdir(data_dir):
        path = os.path.join(data_dir, item)
        if os.path.isdir(path):
            size = sum(os.path.getsize(os.path.join(r, f)) for r, d, fs in os.walk(path) for f in fs)
            count = sum(len(fs) for r, d, fs in os.walk(path))
            print(f"Subdir: {item} - {size / 1024 / 1024:.2f} MB ({count} files)")

if __name__ == "__main__":
    check_data()
