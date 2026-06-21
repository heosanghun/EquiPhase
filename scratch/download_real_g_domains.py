import os
import urllib.request

AA_MAP = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C',
    'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I',
    'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P',
    'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'
}

def download_pdb(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = f"data/pdbs/{pdb_id}.pdb"
    os.makedirs("data/pdbs", exist_ok=True)
    try:
        print(f"Downloading PDB {pdb_id}...")
        urllib.request.urlretrieve(url, output_path)
        print("Success.")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

def parse_pdb(pdb_path):
    coords = []
    seq = []
    if not os.path.exists(pdb_path):
        return None, None
        
    first_chain = None
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ENDMDL'):
                break
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                chain_id = line[21].strip()
                if first_chain is None:
                    first_chain = chain_id
                if chain_id != first_chain:
                    continue
                res_name = line[17:20].strip()
                one_letter = AA_MAP.get(res_name, 'X')
                seq.append(one_letter)
                
    if len(seq) == 0:
        return None, None
    return "".join(seq), len(seq)

def main():
    for pdb_id in ["2LHC", "2LHD"]:
        if download_pdb(pdb_id):
            seq, L = parse_pdb(f"data/pdbs/{pdb_id}.pdb")
            print(f"PDB: {pdb_id} | Length: {L} | Seq: {seq}")

if __name__ == "__main__":
    main()
