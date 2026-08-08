import sys

if len(sys.argv) > 2:
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        content = f.read()
    try:
        text = content.decode("utf-16le")
    except Exception:
        text = content.decode("utf-8", errors="ignore")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Converted {src} to {dst}")
