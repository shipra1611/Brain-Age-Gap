"""
Train Pretrained ViT for Brain Age Prediction
Uses DeiT-Small pretrained on ImageNet
Expected: MAE ~6-8 years (much better than from-scratch!)
Run: python 3_train_vit.py
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
from models.vit_model import BrainAgeViT


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def denormalize_ages(ages_normalized, dataset):
    base = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    return np.array(ages_normalized) * base.age_std + base.age_mean


def train_epoch(model, loader, optimizer, criterion, device, dataset):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, ages_norm, _ in pbar:
        images    = images.to(device)
        ages_norm = ages_norm.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = criterion(preds, ages_norm)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(ages_norm.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    preds_yr   = denormalize_ages(all_preds,   dataset)
    targets_yr = denormalize_ages(all_targets, dataset)
    mae = np.mean(np.abs(preds_yr - targets_yr))
    return total_loss / len(loader), mae


def validate(model, loader, criterion, device, dataset):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, ages_norm, _ in tqdm(loader, desc="Validating", leave=False):
            images    = images.to(device)
            ages_norm = ages_norm.to(device)

            preds = model(images)
            loss  = criterion(preds, ages_norm)

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(ages_norm.cpu().numpy())

    preds_yr   = denormalize_ages(all_preds,   dataset)
    targets_yr = denormalize_ages(all_targets, dataset)

    mae = np.mean(np.abs(preds_yr - targets_yr))
    r   = np.corrcoef(preds_yr, targets_yr)[0, 1] if len(preds_yr) > 1 else 0.0
    return total_loss / len(loader), mae, r, preds_yr, targets_yr


def plot_results(history, preds, targets, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Pretrained ViT - Brain Age Results', fontsize=16, fontweight='bold')

    axes[0,0].plot(history['train_mae'], 'b-', label='Train', lw=2)
    axes[0,0].plot(history['val_mae'],   'r-', label='Val',   lw=2)
    best = np.argmin(history['val_mae'])
    axes[0,0].axvline(x=best, color='g', linestyle='--', alpha=0.5, label=f'Best={best+1}')
    axes[0,0].set(xlabel='Epoch', ylabel='MAE (years)', title='Mean Absolute Error')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(history['train_loss'], 'b-', label='Train', lw=2)
    axes[0,1].plot(history['val_loss'],   'r-', label='Val',   lw=2)
    axes[0,1].set(xlabel='Epoch', ylabel='Loss', title='Training Loss')
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(history['learning_rates'], 'g-', lw=2)
    axes[0,2].set(xlabel='Epoch', ylabel='Learning Rate', title='LR Schedule')
    axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(history['val_r'], 'purple', lw=2)
    axes[1,0].axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Target r>0.9')
    axes[1,0].set(xlabel='Epoch', ylabel='Correlation (r)',
                  title='Validation Correlation', ylim=[0,1])
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].scatter(targets, preds, alpha=0.6, s=40, edgecolors='k', lw=0.3)
    axes[1,1].plot([min(targets), max(targets)], [min(targets), max(targets)],
                   'r--', lw=2, label='Perfect')
    mae_v = np.mean(np.abs(np.array(preds)-np.array(targets)))
    corr  = np.corrcoef(preds, targets)[0,1]
    axes[1,1].set(xlabel='Chronological Age', ylabel='Predicted Age',
                  title=f'Predictions\nMAE={mae_v:.2f} yrs, r={corr:.3f}')
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    residuals = np.array(preds) - np.array(targets)
    axes[1,2].scatter(targets, residuals, alpha=0.6, s=40, edgecolors='k', lw=0.3)
    axes[1,2].axhline(y=0,  color='r',      linestyle='--', lw=2)
    axes[1,2].axhline(y=5,  color='orange', linestyle='--', alpha=0.5)
    axes[1,2].axhline(y=-5, color='orange', linestyle='--', alpha=0.5)
    axes[1,2].set(xlabel='Chronological Age', ylabel='Error (years)', title='Residuals')
    axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {save_path}")
    plt.close()


def main():
    set_seed(42)

    print("=" * 70)
    print(" " * 12 + "PRETRAINED VIT - BRAIN AGE TRAINING")
    print("=" * 70)

    # Device
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"\n✅ GPU: {torch.cuda.get_device_name()}")
        BATCH_SIZE = 32   # Pretrained model → larger batch fine
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        print(f"\n✅ Apple Metal GPU")
        BATCH_SIZE = 32
    else:
        DEVICE = torch.device('cpu')
        print(f"\n⚠️  CPU only")
        BATCH_SIZE = 8

    # Hyperparameters
    # Key insight: pretrained model needs LOW LR (fine-tuning not training)
    MAX_EPOCHS   = 100
    BASE_LR      = 1e-4    # LOW LR for fine-tuning pretrained model
    MIN_LR       = 1e-7
    WEIGHT_DECAY = 1e-4
    DROPOUT      = 0.1
    PATIENCE     = 20
    WARMUP       = 5       # Short warmup (already pretrained)

    print(f"\n⚙️  Configuration:")
    print(f"   Device:      {DEVICE}")
    print(f"   Batch size:  {BATCH_SIZE}")
    print(f"   Base LR:     {BASE_LR}  (low - fine-tuning)")
    print(f"   Warmup:      {WARMUP} epochs")
    print(f"   Max epochs:  {MAX_EPOCHS}")
    print(f"   Patience:    {PATIENCE}")

    # Datasets
    print(f"\n{'='*70}")
    print("Loading Datasets")
    print("=" * 70)

    train_full = BrainAgeDataset('data/processed', augment=True,  use_3_slices=True)
    val_full   = BrainAgeDataset('data/processed', augment=False, use_3_slices=True)

    np.random.seed(42)
    indices    = np.random.permutation(len(train_full))
    train_size = int(0.8 * len(indices))

    train_dataset = Subset(train_full, indices[:train_size])
    val_dataset   = Subset(val_full,   indices[train_size:])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print(f"\n   Training:   {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")

    # Model
    print(f"\n{'='*70}")
    print("Building Pretrained ViT")
    print("=" * 70)

    model = BrainAgeViT(dropout=DROPOUT).to(DEVICE)

    # Use differential learning rates
    # Backbone (pretrained): very low LR
    # Head (new):            normal LR
    backbone_params = [p for n, p in model.named_parameters()
                       if 'head' not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if 'head' in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': BASE_LR / 10},  # 1e-5 for backbone
        {'params': head_params,     'lr': BASE_LR}        # 1e-4 for head
    ], weight_decay=WEIGHT_DECAY)

    criterion = nn.MSELoss()

    # LR scheduler: warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < WARMUP:
            return (epoch + 1) / WARMUP
        progress = (epoch - WARMUP) / (MAX_EPOCHS - WARMUP)
        return max(MIN_LR / BASE_LR,
                   0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print(f"\n{'='*70}")
    print("Training (Fine-tuning pretrained ViT)")
    print("=" * 70)

    best_val_mae     = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae':  [], 'val_mae':  [],
        'val_r':      [], 'learning_rates': []
    }

    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{MAX_EPOCHS}")
        print(f"{'='*70}")

        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer,
            criterion, DEVICE, train_dataset
        )

        val_loss, val_mae, val_r, preds, targets = validate(
            model, val_loader, criterion,
            DEVICE, val_dataset
        )

        scheduler.step()
        current_lr = optimizer.param_groups[1]['lr']  # Head LR

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['val_r'].append(val_r)
        history['learning_rates'].append(current_lr)

        gap = abs(train_mae - val_mae)

        print(f"\n📊 Results:")
        print(f"   Train: Loss={train_loss:.4f}, MAE={train_mae:.2f} years")
        print(f"   Val:   Loss={val_loss:.4f}, MAE={val_mae:.2f} years, r={val_r:.3f}")
        print(f"   Gap:   {gap:.2f} years {'✅' if gap < 2 else '⚠️' if gap < 5 else '❌'}")
        print(f"   LR (head): {current_lr:.2e}")

        if val_mae < best_val_mae - 0.05:
            best_val_mae     = val_mae
            patience_counter = 0

            Path('outputs/models').mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae':              val_mae,
                'val_r':                val_r,
                'history':              history
            }, 'outputs/models/best_vit.pth')

            print(f"   ✅ Best model saved! (MAE: {best_val_mae:.2f})")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

    total_time = time.time() - start_time

    # Final results
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print("=" * 70)

    ckpt = torch.load('outputs/models/best_vit.pth', map_location=DEVICE,weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    _, val_mae, val_r, final_preds, final_targets = validate(
        model, val_loader, criterion, DEVICE, val_dataset
    )

    residuals = np.array(final_preds) - np.array(final_targets)

    print(f"\n🏆 Pretrained ViT Results:")
    print(f"   Val MAE:     {val_mae:.2f} years")
    print(f"   Correlation: {val_r:.3f}")
    print(f"   RMSE:        {np.sqrt(np.mean(residuals**2)):.2f} years")
    print(f"   Best Epoch:  {ckpt['epoch']}")
    print(f"   Train Time:  {total_time/60:.1f} minutes")

    cnn_path = Path('outputs/models/best_cnn.pth')
    if cnn_path.exists():
        cnn_ckpt    = torch.load(cnn_path, map_location=DEVICE)
        cnn_mae     = cnn_ckpt['val_mae']
        improvement = cnn_mae - val_mae
        print(f"\n📈 CNN vs Pretrained ViT:")
        print(f"   CNN MAE: {cnn_mae:.2f} years")
        print(f"   ViT MAE: {val_mae:.2f} years")
        if improvement > 0:
            print(f"   ✅ ViT improved by {improvement:.2f} years!")
        else:
            print(f"   ⚠️  CNN still better by {-improvement:.2f} years")
            print(f"   💡 Use ensemble: avg(CNN, ViT) for best results")

    Path('outputs').mkdir(exist_ok=True)
    plot_results(history, final_preds, final_targets, 'outputs/vit_training.png')

    with open('outputs/vit_training_history.json', 'w') as f:
        json.dump({
            'history':            history,
            'best_mae':           float(val_mae),
            'best_r':             float(val_r),
            'best_epoch':         int(ckpt['epoch']),
            'total_time_minutes': float(total_time / 60)
        }, f, indent=2)

    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"   Next: python compare_models.py")


if __name__ == "__main__":
    main()