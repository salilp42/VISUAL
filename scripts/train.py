"""
Training script for NeuroHoloNet.
"""

import os
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

from neuroholonet.models import DRHPNet
from neuroholonet.data import CubeSphereDataset, ComprehensiveAugmentor
from neuroholonet.metrics import NeuroHoloLoss
from neuroholonet.visualization import AttentionVisualizer

class Config:
    def __init__(self):
        self.seed = 0
        self.batch_size = 4
        self.lr = 1e-3
        self.epochs = 5
        self.voxel_size = 32
        self.grad_accum_steps = 2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def apply(self):
        torch.manual_seed(self.seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)

    def save_to_file(self, filepath):
        with open(filepath, 'w') as f:
            for k,v in self.__dict__.items():
                f.write(f"{k}={v}\n")

def train_one_epoch(model, loader, optimizer, criterion, device="cpu", scaler=None, grad_accum_steps=1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    for bidx, (vox, lab) in enumerate(loader):
        vox = vox.to(device, dtype=torch.float)
        lab = lab.to(device)
        
        with autocast(enabled=(scaler is not None)):
            out = model(vox)
            loss = criterion(out, lab, vox)
            
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        if (bidx+1)%grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item()*len(lab)
        preds = out.argmax(dim=1)
        correct += (preds==lab).sum().item()
        total += len(lab)
        
    return total_loss/total, correct/total

@torch.no_grad()
def val_one_epoch(model, loader, criterion, device="cpu", scaler=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for vox, lab in loader:
        vox = vox.to(device, dtype=torch.float)
        lab = lab.to(device)
        
        with autocast(enabled=(scaler is not None)):
            out = model(vox)
            loss = criterion(out, lab, vox)
            
        total_loss += loss.item()*len(lab)
        preds = out.argmax(dim=1)
        correct += (preds==lab).sum().item()
        total += len(lab)
        
    return total_loss/total, correct/total

def main():
    config = Config()
    config.apply()

    # Create results directory
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{now_str}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    # Save config
    config_path = os.path.join(results_dir, "config.txt")
    config.save_to_file(config_path)

    # Setup data
    transform = ComprehensiveAugmentor(rot_prob=0.5, scale_prob=0.3, noise_prob=0.2)
    train_dataset = CubeSphereDataset(n_samples=32, size=config.voxel_size, transform=transform)
    val_dataset = CubeSphereDataset(n_samples=8, size=config.voxel_size, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    model = DRHPNet(voxel_size=config.voxel_size, base_channels=8, d_model=32, num_classes=2).to(config.device)
    print("Model #params:", sum(p.numel() for p in model.parameters()))

    # Setup training
    criterion = NeuroHoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scaler = GradScaler()

    # Training loop
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(config.epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device=config.device, scaler=scaler, grad_accum_steps=config.grad_accum_steps)
        
        val_loss, val_acc = val_one_epoch(
            model, val_loader, criterion, device=config.device, scaler=scaler)
        
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss={tr_loss:.4f}, Acc={tr_acc:.2f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(results_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Best model saved at epoch={epoch+1}, val_acc={val_acc:.2f}")

    # Plot training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "figures", "training_curves.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
