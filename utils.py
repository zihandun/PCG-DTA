import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch_geometric import data as DATA
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


# 1. Dataset

class TestbedDataset(Dataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None, 
                 pre_transform=None, smile_graph=None, smile_tensor=None, target_graph=None, target_key=None):
        self.root = root
        self.dataset = dataset
        self.process(xd, xt, y, smile_graph, smile_tensor, target_graph, target_key)

    def process(self, xd, xt, y, smile_graph, smile_tensor, target_graph, target_key):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        
        unique_drugs = sorted(list(set(xd)))
        unique_targets = sorted(list(set(target_key)))
        
        self.drug2id = {d: i for i, d in enumerate(unique_drugs)}
        self.target2id = {t: i for i, t in enumerate(unique_targets)}
        
        num_drugs = len(unique_drugs)
        num_targets = len(unique_targets)
        
        print(f"Dataset Info: Found {num_drugs} unique drugs and {num_targets} unique targets.")

        self.affinity_matrix = torch.full((num_drugs, num_targets), -1.0, dtype=torch.float)

        Drug_data_list = []
        Target_data_list = []
        data_len = len(xd)

        print("Processing Multimodal Data...")
        
        for i in tqdm(range(data_len), desc="Building Dataset"):
            smiles = xd[i]
            target_seq = xt[i]    # xt
            labels = y[i]
            key = target_key[i]

            c_size, features, edge_index = smile_graph[smiles]
            tar_size, tar_features, tar_edge_index = target_graph[key]
            
            smile_ten = smile_tensor[smiles]

            d_id = self.drug2id[smiles]
            t_id = self.target2id[key]
            self.affinity_matrix[d_id, t_id] = labels

            DrugData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([labels])
            )
            DrugData.smiles = torch.LongTensor([smile_ten])
            DrugData.__setitem__('c_size', torch.LongTensor([c_size]))
            DrugData.d_idx = torch.tensor([d_id], dtype=torch.long)

            TargetData = DATA.Data(
                x=torch.Tensor(tar_features), 
                edge_index=torch.LongTensor(tar_edge_index).transpose(1, 0),
                y=torch.FloatTensor([labels])
            )
            if isinstance(target_seq, torch.Tensor):
                TargetData.target = target_seq.unsqueeze(0)
            else:
                TargetData.target = torch.LongTensor([target_seq])
                
            TargetData.__setitem__('tar_size', torch.LongTensor([tar_size]))
            TargetData.t_idx = torch.tensor([t_id], dtype=torch.long)

            Drug_data_list.append(DrugData)
            Target_data_list.append(TargetData)

        self.DrugData = Drug_data_list
        self.TargetData = Target_data_list
        
        print(f"Success: Matrix {self.affinity_matrix.shape} built. Multimodal data ready.")

    def __len__(self):
        return len(self.DrugData)

    def __getitem__(self, idx):
        return self.DrugData[idx], self.TargetData[idx]



def collate(data_list):

    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB

def compute_cl_loss(z1, z2, temperature=0.07):
    batch_size = z1.size(0)
    if batch_size < 2: return torch.tensor(0.0).to(z1.device)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(batch_size).to(z1.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2


def supervised_contrastive_loss(z1, z2, affinity_mask, temperature=0.07):
    """
    z1, z2: [Batch, Dim]
    affinity_mask: [Batch, Batch]
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / temperature
    
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mask_sum = affinity_mask.sum(1)
    mask_sum[mask_sum == 0] = 1.0 # 防止除0，反正分子也是0
    
    mean_log_prob_pos = (affinity_mask * log_prob).sum(1) / mask_sum

    loss = - mean_log_prob_pos * (affinity_mask.sum(1) > 0).float()
    
    return loss.mean()
### Davis 0.25 KIBA 0.10
def train(model, device, train_loader, optimizer, epoch, dataset, cl_beta=0.25, threshold=7.0):
    model.train()
    total_loss = 0
    total_reg = 0
    total_cl = 0
    
    loss_fn = torch.nn.SmoothL1Loss()
    
    global_affinity_matrix = dataset.affinity_matrix.to(device)
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (data_mol, data_pro) in pbar:
        DrugData = data_mol.to(device)
        TargetData = data_pro.to(device)
        targets = DrugData.y.view(-1).float().to(device)
        
        b_d_idx = DrugData.d_idx  # Shape: [Batch]
        b_t_idx = TargetData.t_idx # Shape: [Batch]

        optimizer.zero_grad()
        
        out, z_drug, z_target = model(DrugData, TargetData)
        
        loss_reg = loss_fn(out.view(-1), targets)

        batch_affinity_labels = global_affinity_matrix[b_d_idx[:, None], b_t_idx[None, :]]
        
        pos_mask = (batch_affinity_labels >= threshold).float()
        
        if pos_mask.sum() > 0:
            loss_cl = supervised_contrastive_loss(z_drug, z_target, pos_mask)
        else:
            loss_cl = torch.tensor(0.0).to(device)

        loss = loss_reg + (cl_beta * loss_cl)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_reg += loss_reg.item()
        total_cl += loss_cl.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'reg': f'{loss_reg.item():.4f}',
            'cl': f'{loss_cl.item():.4f}'
        })

    return total_loss / len(train_loader), total_cl / len(train_loader)

def train_(model, device, train_loader, optimizer, epoch, cl_alpha=0.0, cl_beta=0.25, threshold=7.0):

    model.train()
    total_loss = 0
    total_reg = 0
    total_cl = 0
    
    loss_fn = torch.nn.SmoothL1Loss()
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} Training", leave=False)

    for batch_idx, (data_mol, data_pro) in pbar:
        DrugData = data_mol.to(device)
        TargetData = data_pro.to(device)
        targets = DrugData.y.view(-1).float().to(device) 
        optimizer.zero_grad()
        
        out, z_drug_inter, z_target_inter = model(DrugData, TargetData)
        
        loss_reg = loss_fn(out.view(-1), targets)

        pos_mask = (targets >= threshold)
        loss_inter = torch.tensor(0.0).to(device)
        
        if pos_mask.sum().item() >= 2:
            loss_inter = compute_cl_loss(z_drug_inter[pos_mask], z_target_inter[pos_mask])

        loss = loss_reg + (cl_beta * loss_inter)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_reg += loss_reg.item()
        total_cl += loss_inter.item() if isinstance(loss_inter, torch.Tensor) else loss_inter

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'reg': f'{loss_reg.item():.4f}',
            'cl': f'{loss_inter.item():.4f}'
        })

    avg_loss = total_loss / len(train_loader)
    avg_reg = total_reg / len(train_loader)
    
    return avg_loss, avg_reg



def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output, _, _ = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()



# Metrics

def mse(y, f): return ((y - f)**2).mean(axis=0)

def ci(y, f):

    ind = np.argsort(y)

    y = y[ind]

    f = f[ind]

    i = len(y)-1

    j = i-1

    z = 0.0

    S = 0.0

    while i > 0:

        while j >= 0:

            if y[i] > y[j]:

                z = z+1

                u = f[i] - f[j]

                if u > 0: S = S + 1

                elif u == 0: S = S + 0.5

            j = j - 1

        i = i - 1

        j = i-1

    return S/z