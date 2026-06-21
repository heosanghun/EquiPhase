import torch
import torch.nn as nn
import numpy as np

class ISSTrainer:
    """
    ISSTrainer manages training and validation epochs for the ISS model,
    enforcing orthogonal starting state diversity, tracking basin collapse rates,
    and handling epoch-level logging.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, writer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = writer
        
        # 1. Orthogonal Diversity Initialization
        # Force z_init_proj starts to be mutually orthogonal to span the latent space
        self.initialize_orthogonal_starts()
        
    def initialize_orthogonal_starts(self):
        """
        Applies orthogonal initialization to the model's z_init_proj parameter.
        """
        with torch.no_grad():
            torch.nn.init.orthogonal_(self.model.z_init_proj)
        print("Successfully applied Orthogonal Initialization to z_init_proj parameter.")
        
    def train_step(self, batch):
        self.model.train()
        padded_X, lams, targets_A, targets_B, ddgs, _, mut_indices, padded_X_wt = batch
        
        padded_X = padded_X.to(self.device)
        lams = lams.to(self.device)
        ddgs = ddgs.to(self.device)
        mut_indices = mut_indices.to(self.device)
        padded_X_wt = padded_X_wt.to(self.device)
        
        self.optimizer.zero_grad()
        
        z_star, margins, coords_pred = self.model(padded_X, lams, mut_indices=mut_indices, X_wt_esm=padded_X_wt)
        # Compute baseline margins at lam = 0 to prevent control parameter shortcutting
        _, margins_zero, _ = self.model(padded_X, torch.zeros_like(lams), mut_indices=mut_indices, X_wt_esm=padded_X_wt)
        
        from equiphase.models.losses import MasterpieceLoss
        if isinstance(self.criterion, MasterpieceLoss):
            t_A = targets_A.to(self.device)
            t_B = targets_B.to(self.device)
            loss, loss_dict = self.criterion(coords_pred, t_A, t_B, z_star, margins=margins, delta_delta_g=ddgs)
        else:
            loss, loss_dict = self.criterion(z_star, margins, coords_pred, padded_X, self.model, ddgs, self.model.z_init_last, margins_zero=margins_zero)
        
        loss.backward()
        mutation_params = [p for n, p in self.model.named_parameters() if "mutation_head" in n and p.requires_grad]
        other_params = [p for n, p in self.model.named_parameters() if "mutation_head" not in n and p.requires_grad]
        if len(other_params) > 0:
            torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)
        if len(mutation_params) > 0:
            torch.nn.utils.clip_grad_norm_(mutation_params, max_norm=1.0)
        self.optimizer.step()
        
        # Compute collapse rate: fraction of batch samples where all starts collapse to the same basin
        with torch.no_grad():
            if self.model.num_starts > 1:
                dists = torch.cdist(z_star, z_star, p=2) # (B, K, K)
                max_dists = dists.max(dim=-1)[0].max(dim=-1)[0] # (B,)
                collapse_rate = (max_dists < 1e-3).float().mean().item()
            else:
                collapse_rate = 1.0
        loss_dict["collapse_rate"] = collapse_rate
        
        return loss.item(), loss_dict
        
    def val_step(self, batch):
        self.model.eval()
        padded_X, lams, targets_A, targets_B, ddgs, _, mut_indices, padded_X_wt = batch
        
        padded_X = padded_X.to(self.device)
        lams = lams.to(self.device)
        ddgs = ddgs.to(self.device)
        mut_indices = mut_indices.to(self.device)
        padded_X_wt = padded_X_wt.to(self.device)
        
        with torch.no_grad():
            z_star, margins, coords_pred = self.model(padded_X, lams, mut_indices=mut_indices, X_wt_esm=padded_X_wt)
            # Compute baseline margins at lam = 0 to prevent control parameter shortcutting
            _, margins_zero, _ = self.model(padded_X, torch.zeros_like(lams), mut_indices=mut_indices, X_wt_esm=padded_X_wt)
            
            from equiphase.models.losses import MasterpieceLoss
            if isinstance(self.criterion, MasterpieceLoss):
                t_A = targets_A.to(self.device)
                t_B = targets_B.to(self.device)
                loss, loss_dict = self.criterion(coords_pred, t_A, t_B, z_star, margins=margins, delta_delta_g=ddgs)
            else:
                loss, loss_dict = self.criterion(z_star, margins, coords_pred, padded_X, self.model, ddgs, self.model.z_init_last, margins_zero=margins_zero)
            
            # Compute collapse rate
            if self.model.num_starts > 1:
                dists = torch.cdist(z_star, z_star, p=2) # (B, K, K)
                max_dists = dists.max(dim=-1)[0].max(dim=-1)[0] # (B,)
                collapse_rate = (max_dists < 1e-3).float().mean().item()
            else:
                collapse_rate = 1.0
        loss_dict["collapse_rate"] = collapse_rate
            
        return loss.item(), loss_dict

    def train_epoch(self):
        epoch_losses = []
        epoch_dicts = []
        
        for batch in self.train_loader:
            loss_val, loss_dict = self.train_step(batch)
            epoch_losses.append(loss_val)
            epoch_dicts.append(loss_dict)
            
        # Aggregate loss dictionary
        avg_dict = {}
        for key in epoch_dicts[0].keys():
            avg_dict[key] = np.mean([d[key] for d in epoch_dicts])
            
        return np.mean(epoch_losses), avg_dict
        
    def val_epoch(self):
        epoch_losses = []
        epoch_dicts = []
        
        for batch in self.val_loader:
            loss_val, loss_dict = self.val_step(batch)
            epoch_losses.append(loss_val)
            epoch_dicts.append(loss_dict)
            
        # Aggregate loss dictionary
        avg_dict = {}
        for key in epoch_dicts[0].keys():
            avg_dict[key] = np.mean([d[key] for d in epoch_dicts])
            
        return np.mean(epoch_losses), avg_dict

    def fit(self, epochs):
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(1, epochs + 1):
            train_loss, train_dict = self.train_epoch()
            val_loss, val_dict = self.val_epoch()
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            # Dynamic logging based on loss keys
            if "loss_soft_min" in train_dict:
                loss_str = f"SoftMin: {train_dict['loss_soft_min']:.4f}, Rep: {train_dict['loss_repulsive']:.4f}, Anchor: {train_dict.get('loss_anchor', 0.0):.4f}"
                val_loss_str = f"SoftMin: {val_dict['loss_soft_min']:.4f}, Rep: {val_dict['loss_repulsive']:.4f}, Anchor: {val_dict.get('loss_anchor', 0.0):.4f}"
            else:
                loss_str = f"Gnm: {train_dict.get('L_gnm', 0.0):.4f}, Contact: {train_dict.get('L_contact', 0.0):.4f}, Phys: {train_dict.get('L_phys', 0.0):.4f}, Rep: {train_dict.get('L_repulsive', 0.0):.4f}, Switch: {train_dict.get('L_switch', 0.0):.4f}, Contract: {train_dict.get('L_contract', 0.0):.4f}"
                val_loss_str = f"Gnm: {val_dict.get('L_gnm', 0.0):.4f}, Contact: {val_dict.get('L_contact', 0.0):.4f}, Phys: {val_dict.get('L_phys', 0.0):.4f}, Rep: {val_dict.get('L_repulsive', 0.0):.4f}, Switch: {val_dict.get('L_switch', 0.0):.4f}, Contract: {val_dict.get('L_contract', 0.0):.4f}"
                
            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("CollapseRate/Train", train_dict['collapse_rate'], epoch)
                self.writer.add_scalar("CollapseRate/Val", val_dict['collapse_rate'], epoch)
                for k, v in train_dict.items():
                    if k != "collapse_rate":
                        self.writer.add_scalar(f"Metric_Train/{k}", v, epoch)
                for k, v in val_dict.items():
                    if k != "collapse_rate":
                        self.writer.add_scalar(f"Metric_Val/{k}", v, epoch)
                        
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} ({loss_str}) | Val Loss: {val_loss:.4f} ({val_loss_str}) | Train Collapse: {train_dict['collapse_rate']:.1%} | Val Collapse: {val_dict['collapse_rate']:.1%}")
            
        return history
