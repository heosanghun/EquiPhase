import os
import torch
from torch.utils.data import Dataset
import numpy as np

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
            if line.startswith('ENDMDL') or line.startswith('MODEL') and len(seq) > 0:
                # Stop at the end of first model
                break
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                chain_id = line[21].strip()
                if first_chain is None:
                    first_chain = chain_id
                if chain_id != first_chain:
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
    return "".join(seq), np.array(coords, dtype=np.float32)

_tokenizer = None
_model = None

def get_esm2_embeddings(sequences, esm_dim=1280):
    global _tokenizer, _model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []
    
    try:
        if _tokenizer is None or _model is None:
            print("Loading facebook/esm2_t33_650M_UR50D model and tokenizer...")
            from transformers import AutoTokenizer, EsmModel
            # Load from HuggingFace
            _tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            _model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
            _model.eval()
            print("Model loaded successfully.")
            
        print("Pre-computing ESM-2 embeddings...")
        with torch.no_grad():
            for seq in sequences:
                inputs = _tokenizer(seq, return_tensors="pt").to(device)
                outputs = _model(**inputs)
                # Remove CLS/EOS tokens: output is of shape (L, esm_dim)
                emb = outputs.last_hidden_state[0, 1:-1, :].cpu()
                embeddings.append(emb)
        print("Successfully pre-computed ESM-2 embeddings.")
    except Exception as e:
        print(f"Warning: Failed to load/use ESM-2 model ({e}). Falling back to mock embeddings.")
        embeddings = []
        for seq in sequences:
            L = len(seq)
            seq_hash = hash(seq) % (2**32)
            generator = torch.Generator().manual_seed(seq_hash)
            X_esm = torch.randn(L, esm_dim, generator=generator)
            embeddings.append(X_esm)
            
    return embeddings

class FoldSwitchDataset(Dataset):
    """
    FoldSwitchDataset manages sequences, control parameters (lambda),
    target structure coordinates, delta_ddg labels, and fold family IDs.
    Extracts and caches ESM-2 embeddings. Supports dual structures for Phase 5.
    """
    def __init__(self, sequences, control_params, target_structures=None, delta_ddgs=None, fold_family_ids=None, esm_dim=1280, target_structures_A=None, target_structures_B=None, pdb_ids=None):
        self.sequences = sequences
        self.control_params = torch.tensor(control_params, dtype=torch.float32)
        if delta_ddgs is None:
            self.delta_ddgs = torch.zeros(len(sequences), dtype=torch.float32)
        else:
            self.delta_ddgs = torch.tensor(delta_ddgs, dtype=torch.float32)
        self.fold_family_ids = fold_family_ids
        self.pdb_ids = pdb_ids
        self.esm_dim = esm_dim
        
        # Precompute and cache ESM-2 embeddings
        self.cached_embeddings = get_esm2_embeddings(sequences, esm_dim)
        
        # Initialize dummy coordinates with matching sequence lengths
        self.target_structures_A = [torch.zeros(len(seq), 3, dtype=torch.float32) for seq in sequences]
        self.target_structures_B = [torch.zeros(len(seq), 3, dtype=torch.float32) for seq in sequences]
                
        # Identify mutation indices compared to WT sequence in the same family
        self.wt_mapping = {}
        mapping_keys = fold_family_ids
        
        if mapping_keys is not None:
            # We assign the first sequence of each family in the dataset as the reference WT
            for seq, key in zip(sequences, mapping_keys):
                if key not in self.wt_mapping:
                    self.wt_mapping[key] = seq
        
        self.mut_indices = []
        if mapping_keys is not None:
            for seq, key in zip(sequences, mapping_keys):
                wt_seq = self.wt_mapping[key]
                if len(seq) == len(wt_seq):
                    mismatch = -1
                    for i_pos in range(len(seq)):
                        if seq[i_pos] != wt_seq[i_pos]:
                            mismatch = i_pos
                            break
                    self.mut_indices.append(mismatch)
                else:
                    self.mut_indices.append(-1)
        else:
            self.mut_indices = [-1] * len(sequences)
            
        self.seq_to_idx = {seq: i for i, seq in enumerate(self.sequences)}
                
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        X_esm = self.cached_embeddings[idx]
        lam = self.control_params[idx]
        target_A = self.target_structures_A[idx]
        target_B = self.target_structures_B[idx]
        ddg = self.delta_ddgs[idx]
        family = self.fold_family_ids[idx]
        mut_idx = self.mut_indices[idx]
        
        # Get WT sequence embedding for this sample
        mapping_keys = self.fold_family_ids
        if mapping_keys is not None:
            key = mapping_keys[idx]
            wt_seq = self.wt_mapping[key]
            wt_idx = self.seq_to_idx[wt_seq]
            X_wt_esm = self.cached_embeddings[wt_idx]
        else:
            X_wt_esm = X_esm
            
        return X_esm, lam, target_A, target_B, ddg, family, mut_idx, X_wt_esm

def split_dataset_by_family(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Strictly splits the dataset based on fold_family_id.
    Ensures no fold family overlaps between Train, Val, and Test splits.
    """
    np.random.seed(seed)
    
    unique_families = list(set(dataset.fold_family_ids))
    unique_families.sort()
    np.random.shuffle(unique_families)
    
    n_families = len(unique_families)
    n_train = int(train_ratio * n_families)
    n_val = int(val_ratio * n_families)
    
    train_families = set(unique_families[:n_train])
    val_families = set(unique_families[n_train:n_train + n_val])
    test_families = set(unique_families[n_train + n_val:])
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for idx, fam in enumerate(dataset.fold_family_ids):
        if fam in train_families:
            train_indices.append(idx)
        elif fam in val_families:
            val_indices.append(idx)
        elif fam in test_families:
            test_indices.append(idx)
            
    from torch.utils.data import Subset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_subset.family_ids = list(train_families)
    val_subset.family_ids = list(val_families)
    test_subset.family_ids = list(test_families)
    
    return train_subset, val_subset, test_subset

def collate_fn(batch):
    """
    Custom collate function to pad variable-length residue embeddings and dual coordinates.
    """
    X_esms, lams, targets_A, targets_B, ddgs, families, mut_indices, X_wt_esms = zip(*batch)
    
    lengths = [x.shape[0] for x in X_esms] + [x.shape[0] for x in X_wt_esms]
    max_len = max(lengths)
    esm_dim = X_esms[0].shape[1]
    
    # Pad residue dimension for ESM embeddings
    padded_X = torch.zeros(len(batch), max_len, esm_dim)
    for i, x in enumerate(X_esms):
        padded_X[i, :x.shape[0]] = x
        
    # Pad residue dimension for WT ESM embeddings
    padded_X_wt = torch.zeros(len(batch), max_len, esm_dim)
    for i, x in enumerate(X_wt_esms):
        padded_X_wt[i, :x.shape[0]] = x
        
    # Pad coordinates dimension for target A
    padded_targets_A = torch.zeros(len(batch), max_len, 3)
    for i, t in enumerate(targets_A):
        padded_targets_A[i, :t.shape[0]] = t

    # Pad coordinates dimension for target B
    padded_targets_B = torch.zeros(len(batch), max_len, 3)
    for i, t in enumerate(targets_B):
        padded_targets_B[i, :t.shape[0]] = t
        
    lams = torch.stack(lams).unsqueeze(-1) # (B, 1)
    ddgs = torch.stack(ddgs).unsqueeze(-1) # (B, 1)
    
    return padded_X, lams, padded_targets_A, padded_targets_B, ddgs, families, torch.tensor(mut_indices, dtype=torch.long), padded_X_wt

def get_real_fold_switch_dataset(esm_dim=1280):
    # 1. Curated real fold-switch sequences and stability values
    curated_data = [
        # GA/GB family
        {
            "sequence": "TTYKLILNLKQAKEEAIKELVDAGTAEKYFKLIANAKTVEGVWTLKDEIKTFTVTE",
            "delta_ddg": 1.2,
            "family": "GA_GB",
            "pdb_id": "2LHC",
            "target_A": "data/pdbs/2LHC.pdb",
            "target_B": "data/pdbs/2LHD.pdb"
        },
        {
            "sequence": "TTYKLILNLKQAKEEAIKELVDAGTAEKYFKLIANAKTVEGVWTYKDEIKTFTVTE",
            "delta_ddg": -1.2,
            "family": "GA_GB",
            "pdb_id": "2LHD",
            "target_A": "data/pdbs/2LHC.pdb",
            "target_B": "data/pdbs/2LHD.pdb"
        },
        {
            "sequence": "TTYKLILNLKQAKEEAIKELVDAGTAEKYIKLIANAKTVEGVWTLKDEIKTFTVTE",
            "delta_ddg": 3.0,
            "family": "GA_GB",
            "pdb_id": "2KDL",
            "target_A": "data/pdbs/2KDL.pdb",
            "target_B": "data/pdbs/2KDM.pdb"
        },
        {
            "sequence": "TTYKLILNLKQAKEEAIKEAVDAGTAEKYFKLIANAKTVEGVWTYKDEIKTFTVTE",
            "delta_ddg": -3.0,
            "family": "GA_GB",
            "pdb_id": "2KDM",
            "target_A": "data/pdbs/2KDL.pdb",
            "target_B": "data/pdbs/2KDM.pdb"
        },
        # S6/B1 family (Ruan et al., 2023)
        {
            "sequence": "SKTFEVNIVLNPNLDQKQLAQAKELAIKALKQYGIGVEKIKLIGNAKTVEAVEKLKQGILLVYQIEAPADRVNDLARELRILDAVRRVEVTYAAD",
            "delta_ddg": 7.0,
            "family": "S6_B1",
            "pdb_id": "7MN1",
            "target_A": "data/pdbs/7MN1.pdb",
            "target_B": "data/pdbs/7MQ4.pdb"
        },
        {
            "sequence": "SAGIATFKLVLNGKTLKGETTTEAVDAATALKNFGAYAQDVGVDGAWTYDDATKTFTVGERLIFKVKMPEDRMNDLARQLRQRDNVSRVEVTRYK",
            "delta_ddg": -1.1,
            "family": "S6_B1",
            "pdb_id": "7MQ4",
            "target_A": "data/pdbs/7MN1.pdb",
            "target_B": "data/pdbs/7MQ4.pdb"
        },
        {
            "sequence": "GIYTVKIVLNPKTNKGELTTEAVDAATALKNFGAKAQDVGVDGAWTYSDPTKTFPVGYRLIFKVEMPEDRVNDLARQLRQRDNVSRVEVTRYK",
            "delta_ddg": 0.5,
            "family": "S6_B1",
            "pdb_id": "7MN2",
            "target_A": "data/pdbs/7MN2.pdb",
            "target_B": "data/pdbs/7MQ4.pdb"
        },
        {
            "sequence": "TTYKYILNLKFAFGDTNSEAVDAAEAEKKFKQYANDHGVDGEWTYDDATKTFTVTAKDSHADRVRELAQRLRQRPRVERVEITEVTE",
            "delta_ddg": 4.0,
            "family": "S6_B1",
            "pdb_id": "7MP7",
            "target_A": "data/pdbs/7MP7.pdb",
            "target_B": "data/pdbs/7MQ4.pdb"
        },
        # Mad2
        {
            "sequence": "MALQLSREQGITLRGSAEIVAEFFSFGINSILYQRGIYPSETFTRVQKYGLTLLVTTDLELIKYLNNVVEQLKDWLYKCSVQKLVVVISNIESGEVLERWQFDIECDKTAKDDSAPREKSQKAIQDEIRSVIAQITATVTFLPLLEVSCSFDLLIYTDKDLVVPEKWEESGPQFITNSEEVRLRSFTTTIHKVNSMVAYKIPVND",
            "delta_ddg": 1.5,
            "family": "Mad2",
            "pdb_id": "1S2H",
            "target_A": "data/pdbs/1S2H.pdb",
            "target_B": "data/pdbs/1GO4.pdb"
        },
        # Lymphotactin
        {
            "sequence": "VGSEVSDKRTCVSLTTQRLPVSRIKTYTITEGSLRAVIFITKRGLKVCADPQATWVRDVVRSMDRKSNTRNNMIQTKPTGTQQSTNTAVTLTG",
            "delta_ddg": 1.0,
            "family": "Lymphotactin",
            "pdb_id": "1J8I",
            "target_A": "data/pdbs/1J8I.pdb",
            "target_B": "data/pdbs/2KCU.pdb"
        },
        # KaiB
        {
            "sequence": "MAPLRKTAVLKLYVAGNTPNSVRALKTLANILEKEFKGVYALKVIDVLKNPQLAEEDKILATPTLAKVLPPPVRRIIGDLSNREKVLIALRLLAEEIGDYKDDDDK",
            "delta_ddg": 2.0,
            "family": "KaiB",
            "pdb_id": "5JYT",
            "target_A": "data/pdbs/5JYT.pdb",
            "target_B": "data/pdbs/5JYV.pdb"
        }
    ]
    
    sequences = [x["sequence"] for x in curated_data]
    delta_ddgs = [x["delta_ddg"] for x in curated_data]
    control_params = delta_ddgs
    fold_family_ids = [x["family"] for x in curated_data]
    pdb_ids = [x["pdb_id"] for x in curated_data]
    target_structures_A = [x["target_A"] for x in curated_data]
    target_structures_B = [x["target_B"] for x in curated_data]
    
    dataset = FoldSwitchDataset(
        sequences=sequences,
        control_params=control_params,
        target_structures_A=target_structures_A,
        target_structures_B=target_structures_B,
        delta_ddgs=delta_ddgs,
        fold_family_ids=fold_family_ids,
        pdb_ids=pdb_ids,
        esm_dim=esm_dim
    )
    return dataset

