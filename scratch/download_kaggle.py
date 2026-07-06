import kagglehub

# Download latest version
path = kagglehub.competition_download('wbc-bench-2026')

print("Path to competition files:", path)
