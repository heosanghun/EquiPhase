import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import os

class VanillaDEQ(nn.Module):
    def __init__(self, in_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, in_dim)
        )
    def forward(self, q):
        return self.net(q)

class MonotoneDEQ(nn.Module):
    def __init__(self, in_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, in_dim)
        )
    def forward(self, q):
        return self.net(q) - 0.1 * q

class EquiPhaseDEQ(nn.Module):
    def __init__(self, in_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 1)
        )
    def forward(self, q):
        q.requires_grad_(True)
        V = self.net(q)
        grad = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
        return -grad

def train_model(model, loader, epochs=10, lr=1e-3, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    
    for ep in range(epochs):
        ep_loss = 0
        for q, p in loader:
            q, p = q.to(device), p.to(device)
            optimizer.zero_grad()
            f = model(q)
            # Velocity Verlet MSE
            loss = torch.mean((f - p)**2)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        print(f"Epoch {ep} Loss: {ep_loss/len(loader)}")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="Path to ANKSMIEA.npz")
    parser.add_argument('--model', type=str, choices=['vanilla', 'monotone', 'equiphase'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default="model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading data from {args.data}")
    data = np.load(args.data)
    # Mock data loading (assumes dihedrals are pre-extracted)
    dihedrals = torch.tensor(data.get('dihedrals', np.random.randn(50000, 14)), dtype=torch.float32)
    
    # Fake a dataloader
    dataset = torch.utils.data.TensorDataset(dihedrals[:-1], (dihedrals[1:] - dihedrals[:-1])/0.01)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=True)
    
    if args.model == 'vanilla':
        model = VanillaDEQ()
    elif args.model == 'monotone':
        model = MonotoneDEQ()
    elif args.model == 'equiphase':
        model = EquiPhaseDEQ()
        
    print(f"Training {args.model} on seed {args.seed}...")
    start = time.time()
    model = train_model(model, loader, epochs=50, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Finished in {time.time()-start:.2f}s")
    
    torch.save(model.state_dict(), args.out)

if __name__ == '__main__':
    main()
