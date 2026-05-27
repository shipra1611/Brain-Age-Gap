"""
Train CNN for Brain Age Prediction
Run: python 2_train_cnn.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

from utils.dataset import BrainAgeDataset
from models.cnn_model import BrainAgeCNN

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def denormalize_ages(ages_normalized, dataset):
    """Convert normalized ages back to years"""
    # Get base dataset if wrapped in Subset
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    
    ages_years = np.array(ages_normalized) * base_dataset.age_std + base_dataset.age_mean
    return ages_years

def train_epoch(model, loader, optimizer, criterion, device, dataset):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds_norm = []
    all_targets_norm = []
    
    for images, ages_norm, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        ages_norm = ages_norm.to(device)
        
        # Forward
        optimizer.zero_grad()
        preds_norm = model(images)
        loss = criterion(preds_norm, ages_norm)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        all_preds_norm.extend(preds_norm.detach().cpu().numpy())
        all_targets_norm.extend(ages_norm.cpu().numpy())
    
    # Denormalize for metrics
    preds_years = denormalize_ages(all_preds_norm, dataset)
    targets_years = denormalize_ages(all_targets_norm, dataset)
    
    mae = np.mean(np.abs(preds_years - targets_years))
    
    return total_loss / len(loader), mae

def validate(model, loader, criterion, device, dataset):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds_norm = []
    all_targets_norm = []
    
    with torch.no_grad():
        for images, ages_norm, _ in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            ages_norm = ages_norm.to(device)
            
            preds_norm = model(images)
            loss = criterion(preds_norm, ages_norm)
            
            total_loss += loss.item()
            all_preds_norm.extend(preds_norm.cpu().numpy())
            all_targets_norm.extend(ages_norm.cpu().numpy())
    
    # Denormalize
    preds_years = denormalize_ages(all_preds_norm, dataset)
    targets_years = denormalize_ages(all_targets_norm, dataset)
    
    mae = np.mean(np.abs(preds_years - targets_years))
    r = np.corrcoef(preds_years, targets_years)[0, 1] if len(preds_years) > 1 else 0.0
    
    return total_loss / len(loader), mae, r, preds_years, targets_years

def plot_training_results(history, preds, targets, save_path):
    """Plot training history and predictions"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # MAE
    axes[0, 0].plot(history['train_mae'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(history['val_mae'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MAE (years)')
    axes[0, 0].set_title('Mean Absolute Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overfitting gap
    gap = [abs(t - v) for t, v in zip(history['train_mae'], history['val_mae'])]
    axes[0, 2].plot(gap, 'orange', linewidth=2)
    axes[0, 2].axhline(y=2, color='g', linestyle='--', label='Target < 2yr')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('|Train - Val| MAE')
    axes[0, 2].set_title('Overfitting Gap')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Correlation
    axes[1, 0].plot(history['val_r'], 'purple', linewidth=2, marker='o', markersize=3)
    axes[1, 0].axhline(y=0.9, color='g', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Correlation (r)')
    axes[1, 0].set_title('Validation Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Scatter: Predicted vs Actual
    axes[1, 1].scatter(targets, preds, alpha=0.5, s=30)
    axes[1, 1].plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
    axes[1, 1].set_xlabel('Chronological Age (years)')
    axes[1, 1].set_ylabel('Predicted Age (years)')
    axes[1, 1].set_title(f'Predictions (r={np.corrcoef(preds, targets)[0,1]:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Residuals
    residuals = np.array(preds) - np.array(targets)
    axes[1, 2].scatter(targets, residuals, alpha=0.5, s=30)
    axes[1, 2].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 2].axhline(y=5, color='orange', linestyle='--', alpha=0.5)
    axes[1, 2].axhline(y=-5, color='orange', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Chronological Age (years)')
    axes[1, 2].set_ylabel('Error (years)')
    axes[1, 2].set_title('Residual Plot')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {save_path}")

def main():
    set_seed(42)
    
    print("=" * 70)
    print(" " * 20 + "BRAIN AGE CNN TRAINING")
    print("=" * 70)
    
    # Hyperparameters
    BATCH_SIZE = 16
    MAX_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.4
    PATIENCE = 20
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n⚙️  Configuration:")
    print(f"   Device: {DEVICE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Weight decay: {WEIGHT_DECAY}")
    print(f"   Dropout: {DROPOUT}")
    print(f"   Patience: {PATIENCE}")
    
    # Load datasets
    print(f"\n{'='*70}")
    print("Loading Datasets")
    print("=" * 70)
    
    train_dataset_full = BrainAgeDataset('data/processed', augment=True)
    val_dataset_full = BrainAgeDataset('data/processed', augment=False)
    
    # Train/val split
    np.random.seed(42)
    indices = np.random.permutation(len(train_dataset_full))
    train_size = int(0.8 * len(indices))
    
    train_dataset = Subset(train_dataset_full, indices[:train_size])
    val_dataset = Subset(val_dataset_full, indices[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"\n   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    
    # Create model
    print(f"\n{'='*70}")
    print("Building Model")
    print("=" * 70)
    
    model = BrainAgeCNN(pretrained=True, dropout=DROPOUT, num_input_channels=3).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n   Model: ResNet-18 CNN")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Training setup
    criterion = nn.MSELoss()  # MSE for normalized ages
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    # Training loop
    print(f"\n{'='*70}")
    print("Training")
    print("=" * 70)
    
    best_val_mae = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'val_r': []}
    
    start_time = time.time()
    
    for epoch in range(MAX_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{MAX_EPOCHS}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, DEVICE, train_dataset)
        
        # Validate
        val_loss, val_mae, val_r, preds, targets = validate(model, val_loader, criterion, DEVICE, val_dataset)
        
        scheduler.step(val_mae)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['val_r'].append(val_r)
        
        gap = abs(train_mae - val_mae)
        lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Train: Loss={train_loss:.4f}, MAE={train_mae:.2f} years")
        print(f"   Val:   Loss={val_loss:.4f}, MAE={val_mae:.2f} years, r={val_r:.3f}")
        print(f"   Gap: {gap:.2f} years {'✅' if gap < 2 else '⚠️' if gap < 5 else '❌'}")
        print(f"   LR: {lr:.2e}")
        
        # Save best model
        if val_mae < best_val_mae - 0.05:
            best_val_mae = val_mae
            patience_counter = 0
            
            Path('outputs/models').mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_r': val_r,
                'history': history
            }, 'outputs/models/best_cnn.pth')
            
            print(f"   ✅ Best model saved! (MAE: {best_val_mae:.2f})")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print("=" * 70)
    
    checkpoint = torch.load('outputs/models/best_cnn.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_mae, val_r, final_preds, final_targets = validate(model, val_loader, criterion, DEVICE, val_dataset)
    
    print(f"\n🏆 Best Model:")
    print(f"   Val MAE: {val_mae:.2f} years")
    print(f"   Correlation: {val_r:.3f}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Training time: {total_time/60:.1f} minutes")
    
    # Plot results
    Path('outputs').mkdir(exist_ok=True)
    plot_training_results(history, final_preds, final_targets, 'outputs/cnn_training.png')
    
    # Save history
    with open('outputs/training_history.json', 'w') as f:
        json.dump({
            'history': history,
            'best_mae': float(val_mae),
            'best_r': float(val_r),
            'best_epoch': int(checkpoint['epoch'])
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()