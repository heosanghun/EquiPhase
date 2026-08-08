import re

path = r"C:\Project\EquiPhase\paper1_manuscript_draft.md"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

print("File Length:", len(content), "chars")
matches = re.findall(r'[A-Z][a-zA-Z\s,]+(?:\set\sal\.|,\s[A-Z]\.)?\s*\(\d{4}\)', content)
print("Found Citations in Text:")
for m in set(matches):
    print(" -", m)
