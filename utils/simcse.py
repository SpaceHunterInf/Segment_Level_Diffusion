import torch
import torch.nn as nn
import torch.nn.functional as F

def get_simcse_loss(org, sim, con, pooler='avg', temp=0.05, hard_negative_weight=1):
    if sim is None and con is None:
        return 0
    
    cos_sim = nn.CosineSimilarity(dim=-1)
    batch_size = org.shape[0]
    
    # Pooling and normalization
    if pooler == 'avg':
        org_pooled = F.normalize(org.mean(dim=1), p=2, dim=-1)
        if sim is not None:
            sim_pooled = F.normalize(sim.mean(dim=1), p=2, dim=-1)
        if con is not None:
            con_pooled = F.normalize(con.mean(dim=1), p=2, dim=-1)
    else:
        raise NotImplementedError
    
    if sim_pooled is not None:
        # For supervised SimCSE, compare org with sim
        pos_cos = cos_sim(org_pooled.unsqueeze(1), sim_pooled.unsqueeze(0)) / temp
        
        if con_pooled is not None:
            # Add hard negatives if provided
            con_cos = cos_sim(org_pooled.unsqueeze(1), con_pooled.unsqueeze(0)) / temp
            base = torch.cat([pos_cos, con_cos], 1)
        else:
            base = pos_cos
            
        labels = torch.arange(batch_size).long().to(org.device)
        loss_fct = nn.CrossEntropyLoss()
        
        if con_pooled is not None:
            # Apply weights for hard negatives
            weights = torch.tensor(
                [[0.0] * (base.size(-1) - con_cos.size(-1)) + [0.0] * i + [hard_negative_weight] + [0.0] * (con_cos.size(-1) - i - 1) 
                 for i in range(con_cos.size(-1))]).to(org.device)
            base += weights
        
        loss = loss_fct(base, labels)

        return loss
    
    return 0