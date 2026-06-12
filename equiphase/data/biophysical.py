import numpy as np

# Kyte-Doolittle hydropathy values
KD_VALUES = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def compute_fcr_ncpr(seq):
    if len(seq) == 0:
        return 0.0, 0.0
    pos = sum(1 for aa in seq if aa in 'KR')
    neg = sum(1 for aa in seq if aa in 'DE')
    fcr = (pos + neg) / len(seq)
    ncpr = (pos - neg) / len(seq)
    return fcr, ncpr

def compute_scd(seq):
    L = len(seq)
    if L == 0:
        return 0.0
    
    # Extract charges and indices
    charges = []
    idxs = []
    for idx, aa in enumerate(seq):
        if aa in 'KR':
            charges.append(1.0)
            idxs.append(idx)
        elif aa in 'DE':
            charges.append(-1.0)
            idxs.append(idx)
            
    scd_val = 0.0
    n_charges = len(charges)
    for i in range(n_charges):
        q_i = charges[i]
        pos_i = idxs[i]
        for j in range(i + 1, n_charges):
            q_j = charges[j]
            pos_j = idxs[j]
            scd_val += q_i * q_j * np.sqrt(pos_j - pos_i)
            
    return scd_val / L

def compute_sigma2(charges, w, ncpr):
    L = len(charges)
    if L < w:
        return 0.0
    sums = np.convolve(charges, np.ones(w), 'valid')
    fractions = sums / w
    return np.mean((fractions - ncpr) ** 2)

def compute_kappa(seq, w=5):
    L = len(seq)
    if L < w or L == 0:
        return 0.0
    
    # 1. Build charge array
    charges = []
    for aa in seq:
        if aa in 'KR':
            charges.append(1.0)
        elif aa in 'DE':
            charges.append(-1.0)
        else:
            charges.append(0.0)
    charges = np.array(charges)
    
    pos_count = sum(1 for val in charges if val == 1.0)
    neg_count = sum(1 for val in charges if val == -1.0)
    neut_count = L - pos_count - neg_count
    
    ncpr = (pos_count - neg_count) / L
    
    # Compute sequence sigma^2
    sigma2 = compute_sigma2(charges, w, ncpr)
    
    # Construct fully segregated charges for max_sigma2
    # Segregated: all positives at one end, all negatives at other, neutrals in between
    seg_charges = [1.0] * pos_count + [0.0] * neut_count + [-1.0] * neg_count
    seg_charges = np.array(seg_charges)
    max_sigma2 = compute_sigma2(seg_charges, w, ncpr)
    
    if max_sigma2 <= 1e-8:
        return 0.0
    return float(sigma2 / max_sigma2)

def compute_aromatic_fraction(seq):
    if len(seq) == 0:
        return 0.0
    return sum(1 for aa in seq if aa in 'FYWH') / len(seq)

def compute_sticker_spacer(seq):
    if len(seq) == 0:
        return 0.0, 0.0
    stickers = sum(1 for aa in seq if aa in 'FYWR')
    spacers = sum(1 for aa in seq if aa in 'GSQNPTA')
    return stickers / len(seq), spacers / len(seq)

def compute_motifs(seq):
    if len(seq) == 0:
        return 0
    # Count RG and GR motifs
    rg = seq.count('RG')
    gr = seq.count('GR')
    return rg + gr

def compute_hydropathy(seq):
    if len(seq) == 0:
        return 0.0
    vals = [KD_VALUES.get(aa, 0.0) for aa in seq]
    return np.mean(vals)

def extract_biophysical_features(seq):
    if not isinstance(seq, str):
        seq = ""
    fcr, ncpr = compute_fcr_ncpr(seq)
    scd = compute_scd(seq)
    kappa = compute_kappa(seq)
    aromatic = compute_aromatic_fraction(seq)
    sticker, spacer = compute_sticker_spacer(seq)
    motifs = compute_motifs(seq)
    hydro = compute_hydropathy(seq)
    
    return np.array([
        len(seq),
        fcr,
        ncpr,
        scd,
        kappa,
        aromatic,
        sticker,
        spacer,
        motifs,
        hydro
    ], dtype=np.float32)

BIOPHYSICAL_FEATURE_NAMES = [
    "length", "fcr", "ncpr", "scd", "kappa", "aromatic", "sticker", "spacer", "motifs", "hydropathy"
]
