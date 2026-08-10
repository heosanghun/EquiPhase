import os, glob

replacements = {
    r'C:\Users\wwwhu': '/home/user',
    r'C:\\Users\\wwwhu': '/home/user',
    r'C:\Project\EquiPhase': '/home/user/EquiPhase',
    r'C:\\Project\\EquiPhase': '/home/user/EquiPhase',
    'wwwhu': 'user'
}

for ext in ['*.py', '*.txt', '*.log', '*.md']:
    for fpath in glob.glob('**/' + ext, recursive=True):
        for enc in ['utf-8', 'utf-16']:
            try:
                with open(fpath, 'r', encoding=enc) as f:
                    content = f.read()
                orig = content
                for k, v in replacements.items():
                    content = content.replace(k, v)
                if content != orig:
                    with open(fpath, 'w', encoding=enc) as f:
                        f.write(content)
                    print(f'Anonymized {fpath} ({enc})')
                break # if read successful, break encoding loop
            except Exception as e:
                pass
