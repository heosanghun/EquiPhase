import os

AA_MAP = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
    'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I',
    'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
    'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
}

def parse_pdb(pdb_path):
    coords = []
    seq = []
    if not os.path.exists(pdb_path):
        return None, None
        
    first_chain = None
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ENDMDL'):
                # Stop at the first model
                break
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                chain_id = line[21].strip()
                if first_chain is None:
                    first_chain = chain_id
                if chain_id != first_chain:
                    # Only parse the first chain
                    continue
                res_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                one_letter = AA_MAP.get(res_name, 'X')
                seq.append(one_letter)
                coords.append([x, y, z])
                
    if len(seq) == 0:
        return None, None
    return "".join(seq), len(seq)

def main():
    pdbs_dir = "data/pdbs"
    files = sorted(os.listdir(pdbs_dir))
    for f in files:
        if f.endswith(".pdb") and not f.startswith("pdb_"):
            pdb_path = os.path.join(pdbs_dir, f)
            seq, L = parse_pdb(pdb_path)
            if seq:
                print(f"PDB: {f} | Length: {L} | Seq (first 80): {seq[:80]}")

if __name__ == "__main__":
    main()
