import pandas as pd
import numpy as np
import os
import glob
import torch
from rdkit import Chem
from tqdm import tqdm
import networkx as nx
from collections import Counter
import shutil

from utils import TestbedDataset

def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14, 'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13, 'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36, 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21, 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17, 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13, 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00, 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00, 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100, 'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7, 'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99, 'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5, 'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0, 1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table.get(residue, 0), res_pka_table.get(residue, 0), res_pkb_table.get(residue, 0), res_pkx_table.get(residue, 0),
                     res_pl_table.get(residue, 0), res_hydrophobic_ph2_table.get(residue, 0), res_hydrophobic_ph7_table.get(residue, 0)]
    return np.array(res_property1 + res_property2)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        curr_res = pro_seq[i] if pro_seq[i] in pro_res_table else 'X'
        pro_hot[i,] = one_of_k_encoding(curr_res, pro_res_table)
        pro_property[i,] = residue_features(curr_res)
    return np.concatenate((pro_hot, pro_property), axis=1)

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind.get(ch, 0)
    return X

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
max_seq_len = 1000

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict.get(ch, 0)
    return x

def PSSM_calculation(aln_file, pro_seq):
    pro_res_table_local = 'ACDEFGHIKLMNPQRSTVWYX'
    aa_map = {res: i for i, res in enumerate(pro_res_table_local)}
    seq_len = len(pro_seq)
    pfm_mat = np.zeros((len(pro_res_table_local), seq_len), dtype=np.float32)
    
    if not os.path.exists(aln_file): return pfm_mat

    with open(aln_file, 'r') as f:
        lines = f.readlines()
    
    valid_lines = [line.strip() for line in lines if not line.startswith('>')]
    if not valid_lines: return pfm_mat
        
    transposed_cols = zip(*[line[:seq_len] for line in valid_lines])
    for col_idx, chars in enumerate(transposed_cols):
        counts = Counter(chars)
        for aa, count in counts.items():
            if aa in aa_map:
                pfm_mat[aa_map[aa], col_idx] = count
    
    num_seqs = len(valid_lines)
    pseudocount = 0.8
    bg_freq = 1.0 / 20.0  
    ppm_mat = (pfm_mat + pseudocount * bg_freq) / (float(num_seqs) + pseudocount)
    pssm_mat = np.log2(ppm_mat / bg_freq + 1e-9)
    final_mat = 1.0 / (1.0 + np.exp(-pssm_mat))
    return final_mat

def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    
    target_edge_index = []
    if os.path.exists(contact_file):
        try:
            contact_map = np.load(contact_file)
            index_row, index_col = np.where(contact_map >= 0.5)
            for i, j in zip(index_row, index_col):
                target_edge_index.append([i, j])
        except:
            pass 
    
    pssm = PSSM_calculation(aln_file, target_sequence)
    
    other_feature = seq_feature(target_sequence)
    
    feat = np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)
    
    return len(target_sequence), feat, np.array(target_edge_index)

def process_folder(dataset_name, csv_folder, output_folder='data/processed'):
    print(f"\n{'='*60}")
    print(f"Start processing dataset: {dataset_name}")
    print(f"Source Folder: {csv_folder}")
    print(f"{'='*60}")

    csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
    if not csv_files:
        print(f"Warning: No csv files found in {csv_folder}")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    aln_path = f'data/{dataset_name}/aln'
    contact_path = f'data/{dataset_name}/pconsc4'
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file).replace('.csv', '')
        save_path = os.path.join(output_folder, f'{dataset_name}_{filename}.data')
        
        cache_name = f"{dataset_name}_{filename}.pt" # PyG 默认生成的缓存名
        processed_file_path = os.path.join('data/processed', cache_name)
        
        if os.path.exists(processed_file_path):
            os.remove(processed_file_path)
            print(f"   [Cache Clean] Removed old cache: {processed_file_path}")

        print(f"-> Processing {filename} ...")
        
        df = pd.read_csv(csv_file)
        req_cols = ['compound_iso_smiles', 'target_sequence', 'target_key', 'affinity']
        if not all(col in df.columns for col in req_cols):
            print(f"   [Skip] Missing columns in {csv_file}")
            continue

        smile_graph = {}
        smile_tensor = {}
        target_graph = {}

        unique_smiles = df['compound_iso_smiles'].unique()
        for smile in tqdm(unique_smiles, desc="   Building Drug Graphs", leave=False):
            if smile not in smile_graph:
                g = smile_to_graph(smile)
                if g:
                    smile_graph[smile] = g
                    smile_tensor[smile] = label_smiles(smile, 100, CHARISOSMISET)

        unique_targets = df[['target_key', 'target_sequence']].drop_duplicates(subset=['target_key'])
        for _, row in tqdm(unique_targets.iterrows(), total=len(unique_targets), desc="   Building Target Graphs", leave=False):
            key, seq = row['target_key'], row['target_sequence']
            if key not in target_graph:
                target_graph[key] = target_to_graph(key, seq, contact_path, aln_path)

        valid_mask = df['compound_iso_smiles'].isin(smile_graph) & df['target_key'].isin(target_graph)
        df_clean = df[valid_mask].reset_index(drop=True)
        
        if len(df_clean) < len(df):
            print(f"   Filtered {len(df) - len(df_clean)} invalid samples.")

        dataset = TestbedDataset(
            root='data', 
            dataset=f"{dataset_name}_{filename}", 
            xd=df_clean['compound_iso_smiles'].tolist(),
            xt=np.array([seq_cat(s) for s in df_clean['target_sequence']]),
            y=df_clean['affinity'].values,
            smile_graph=smile_graph,
            smile_tensor=smile_tensor,
            target_graph=target_graph,
            target_key=df_clean['target_key'].values
        )

        torch.save(dataset, save_path)
        print(f"   Saved to: {save_path} (Size: {len(dataset)})")

if __name__ == "__main__":
    if os.path.exists('data/processed'):
        print("[Init] Clearing data/processed folder to ensure clean state...")
        shutil.rmtree('data/processed')
        os.makedirs('data/processed')

    datasets_to_process = [
        ('davis', './data/csv/davis'),
        ('kiba', './data/csv/kiba'),
        ('exdataset', './data/csv/exdataset')
    ]

    for d_name, csv_dir in datasets_to_process:
        if os.path.exists(csv_dir):
            process_folder(d_name, csv_dir)
        else:
            print(f"Directory not found: {csv_dir}")