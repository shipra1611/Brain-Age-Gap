"""
Train Vision Transformer for Brain Age Prediction
FIXED: Removed CosineAnnealingWarmRestarts, added warmup,
       fixed architecture, proper LR scheduling
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

# ============================================================
# WARMUP + COSINE DECAY SCHEDULER (no restarts)
# ============================================================
class WarmupCosineScheduler:
    """
    Linear warmup then cosine decay
    Much more stable than CosineAnnealingWarmRestarts
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine decay (no restart)
            progress = (self.current_epoch - self.warmup_epochs) / \
                       (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def denormalize_ages(ages_normalized, dataset):
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    return np.array(ages_normalized) * base_dataset.age_std + base_dataset.age_mean


def train_epoch(model, loader, optimizer, criterion, device, dataset):
    model.train()
    total_loss = 0
    all_preds_norm = []
    all_targets_norm = []

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, ages_norm, _ in pbar:
        images = images.to(device)
        ages_norm = ages_norm.to(device)

        optimizer.zero_grad()
        preds_norm = model(images)
        loss = criterion(preds_norm, ages_norm)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds_norm.extend(preds_norm.detach().cpu().numpy())
        all_targets_norm.extend(ages_norm.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    preds_years = denormalize_ages(all_preds_norm, dataset)
    targets_years = denormalize_ages(all_targets_norm, dataset)
    mae = np.mean(np.abs(preds_years - targets_years))
    return total_loss / len(loader), mae


def validate(model, loader, criterion, device, dataset):
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

    preds_years = denormalize_ages(all_preds_norm, dataset)
    targets_years = denormalize_ages(all_targets_norm, dataset)

    mae = np.mean(np.abs(preds_years - targets_years))
    r = np.corrcoef(preds_years, targets_years)[0, 1] if len(preds_years) > 1 else 0.0

    return total_loss / len(loader), mae, r, preds_years, targets_years


def plot_results(history, preds, targets, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Vision Transformer - Brain Age Results', fontsize=16, fontweight='bold')

    # MAE
    axes[0, 0].plot(history['train_mae'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(history['val_mae'], 'r-', label='Val', linewidth=2)
    best_epoch = np.argmin(history['val_mae'])
    axes[0, 0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best={best_epoch+1}')
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

    # LR
    axes[0, 2].plot(history['learning_rates'], 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].grid(True, alpha=0.3)

    # Correlation
    axes[1, 0].plot(history['val_r'], 'purple', linewidth=2)
    axes[1, 0].axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Target r>0.9')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Correlation (r)')
    axes[1, 0].set_title('Validation Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])

    # Scatter
    axes[1, 1].scatter(targets, preds, alpha=0.6, s=40, edgecolors='k', linewidth=0.3)
    axes[1, 1].plot([min(targets), max(targets)], [min(targets), max(targets)],
                    'r--', lw=2, label='Perfect')
    axes[1, 1].set_xlabel('Chronological Age (years)')
    axes[1, 1].set_ylabel('Predicted Age (years)')
    corr = np.corrcoef(preds, targets)[0, 1]
    mae_v = np.mean(np.abs(np.array(preds) - np.array(targets)))
    axes[1, 1].set_title(f'Predictions\nMAE={mae_v:.2f} yrs, r={corr:.3f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Residuals
    residuals = np.array(preds) - np.array(targets)
    axes[1, 2].scatter(targets, residuals, alpha=0.6, s=40, edgecolors='k', linewidth=0.3)
    axes[1, 2].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 2].axhline(y=5, color='orange', linestyle='--', alpha=0.5)
    axes[1, 2].axhline(y=-5, color='orange', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Chronological Age (years)')
    axes[1, 2].set_ylabel('Error (years)')
    axes[1, 2].set_title('Residuals')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved: {save_path}")
    plt.close()


def main():
    set_seed(42)

    print("=" * 70)
    print(" " * 15 + "VISION TRANSFORMER TRAINING (FIXED)")
    print("=" * 70)

    # ============================================================
    # DEVICE
    # ============================================================
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✅ GPU: {torch.cuda.get_device_name()} ({gpu_mem:.1f} GB)")
        BATCH_SIZE = 16   # Larger batch = more stable for ViT
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
        print(f"\n✅ Apple Metal GPU")
        BATCH_SIZE = 16
    else:
        DEVICE = torch.device('cpu')
        print(f"\n⚠️  CPU only - will be slow!")
        BATCH_SIZE = 4

    # ============================================================
    # HYPERPARAMETERS (FIXED)
    # ============================================================
    MAX_EPOCHS   = 100
    BASE_LR      = 3e-4    # FIXED: was 5e-4 (slightly lower = more stable)
    MIN_LR       = 1e-6
    WEIGHT_DECAY = 1e-4
    DROPOUT      = 0.1     # FIXED: was 0.2 (lower = better for small dataset)
    PATIENCE     = 25      # FIXED: more patience
    WARMUP_EPOCHS = 10     # NEW: warmup prevents early chaos
    GRAD_CLIP    = 1.0     # FIXED: was 5.0 (tighter = more stable)

    print(f"\n⚙️  Configuration:")
    print(f"   Device:         {DEVICE}")
    print(f"   Batch size:     {BATCH_SIZE}")
    print(f"   Max epochs:     {MAX_EPOCHS}")
    print(f"   Base LR:        {BASE_LR}")
    print(f"   Warmup epochs:  {WARMUP_EPOCHS}")
    print(f"   Dropout:        {DROPOUT}")
    print(f"   Patience:       {PATIENCE}")
    print(f"   Grad clip:      {GRAD_CLIP}")

    # ============================================================
    # DATASETS
    # ============================================================
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
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print(f"\n   Training:   {len(train_dataset)} samples")
    print(f"   Validation: {len(val_dataset)} samples")
    print(f"   Batch size: {BATCH_SIZE}")

    # ============================================================
    # MODEL (SIMPLIFIED - fewer layers = better for small dataset)
    # ============================================================
    print(f"\n{'='*70}")
    print("Building Vision Transformer")
    print("=" * 70)

    model = BrainAgeViT(
        image_size   = 224,
        patch_size   = 16,
        num_channels = 3,
        dim          = 256,    # FIXED: was 384 (simpler = less overfitting)
        depth        = 4,      # FIXED: was 6 (shallower = better for 322 samples)
        heads        = 8,      # FIXED: was 6 (more heads = better attention)
        mlp_dim      = 512,    # FIXED: was 768
        dropout      = DROPOUT
    ).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n   Architecture:")
    print(f"     Patch size:    16x16 → 196 patches")
    print(f"     Embedding dim: 256")
    print(f"     Layers:        4  (reduced for small dataset)")
    print(f"     Heads:         8")
    print(f"     MLP dim:       512")
    print(f"   Total params:    {total_params:,}")
    print(f"   Trainable:       {trainable_params:,}")

    # ============================================================
    # OPTIMIZER + SCHEDULER (FIXED)
    # ============================================================
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(      # FIXED: AdamW better than Adam for ViT
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # FIXED: Simple warmup + cosine decay (no restarts!)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=WARMUP_EPOCHS,
        max_epochs=MAX_EPOCHS,
        base_lr=BASE_LR,
        min_lr=MIN_LR
    )

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    print(f"\n{'='*70}")
    print("Training")
    print(f"   First {WARMUP_EPOCHS} epochs: LR warms up slowly")
    print(f"   Epoch {WARMUP_EPOCHS}+: LR decays with cosine")
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

        # Step scheduler BEFORE training (warmup)
        scheduler.step()
        current_lr = scheduler.get_lr()

        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer,
            criterion, DEVICE, train_dataset
        )

        val_loss, val_mae, val_r, preds, targets = validate(
            model, val_loader, criterion, DEVICE, val_dataset
        )

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
        print(f"   LR:    {current_lr:.2e}")

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
                'history':              history,
                'config': {
                    'dim': 256, 'depth': 4, 'heads': 8,
                    'mlp_dim': 512, 'dropout': DROPOUT
                }
            }, 'outputs/models/best_vit.pth')

            print(f"   ✅ Best model saved! (MAE: {best_val_mae:.2f})")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

    total_time = time.time() - start_time

    # ============================================================
    # FINAL RESULTS
    # ============================================================
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print("=" * 70)

    ckpt = torch.load('outputs/models/best_vit.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

    _, val_mae, val_r, final_preds, final_targets = validate(
        model, val_loader, criterion, DEVICE, val_dataset
    )

    residuals = np.array(final_preds) - np.array(final_targets)

    print(f"\n🏆 Best ViT Model:")
    print(f"   Val MAE:     {val_mae:.2f} years")
    print(f"   Correlation: {val_r:.3f}")
    print(f"   RMSE:        {np.sqrt(np.mean(residuals**2)):.2f} years")
    print(f"   Median AE:   {np.median(np.abs(residuals)):.2f} years")
    print(f"   Best Epoch:  {ckpt['epoch']}")
    print(f"   Train Time:  {total_time/60:.1f} minutes")

    # Compare with CNN
    cnn_path = Path('outputs/models/best_cnn.pth')
    if cnn_path.exists():
        cnn_ckpt    = torch.load(cnn_path, map_location=DEVICE)
        cnn_mae     = cnn_ckpt['val_mae']
        improvement = cnn_mae - val_mae
        print(f"\n📈 CNN vs ViT Comparison:")
        print(f"   CNN MAE: {cnn_mae:.2f} years")
        print(f"   ViT MAE: {val_mae:.2f} years")
        if improvement > 0:
            print(f"   ✅ ViT improved by {improvement:.2f} years!")
        else:
            print(f"   ⚠️  CNN was better by {-improvement:.2f} years")
            print(f"   💡 Consider using CNN or ensemble for best results")

    # Save outputs
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
    print("✅ VISION TRANSFORMER TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Outputs:")
    print(f"   Model:   outputs/models/best_vit.pth")
    print(f"   Plots:   outputs/vit_training.png")
    print(f"   History: outputs/vit_training_history.json")
    print(f"\n🎯 Next step:")
    print(f"   python compare_models.py")


if __name__ == "__main__":
    main()