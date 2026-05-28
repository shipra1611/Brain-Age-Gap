"""
Compare CNN vs Vision Transformer Performance
And create an ensemble model combining both
Run: python compare_models.py
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from utils.dataset import BrainAgeDataset
from models.cnn_model import BrainAgeCNN
from models.vit_model import BrainAgeViT
from torch.utils.data import DataLoader, Subset

def load_dataset():
    """Load validation dataset"""
    val_dataset = BrainAgeDataset('data/processed', augment=False, use_3_slices=True)
    
    # Use same split as training
    np.random.seed(42)
    indices = np.random.permutation(len(val_dataset))
    train_size = int(0.8 * len(indices))
    
    val_indices = indices[train_size:]
    val_dataset_split = Subset(val_dataset, val_indices)
    
    return val_dataset_split, val_dataset

def denormalize_ages(ages_normalized, dataset):
    """Convert normalized ages back to years"""
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    ages_years = np.array(ages_normalized) * base_dataset.age_std + base_dataset.age_mean
    return ages_years

def evaluate_model(model, loader, device, dataset, model_name="Model"):
    """Evaluate single model"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, ages_norm, _ in tqdm(loader, desc=f"Evaluating {model_name}", leave=False):
            images = images.to(device)
            ages_norm = ages_norm.to(device)
            
            preds_norm = model(images)
            
            all_preds.extend(preds_norm.cpu().numpy())
            all_targets.extend(ages_norm.cpu().numpy())
    
    # Denormalize
    preds_years = denormalize_ages(all_preds, dataset)
    targets_years = denormalize_ages(all_targets, dataset)
    
    # Calculate metrics
    mae = np.mean(np.abs(preds_years - targets_years))
    rmse = np.sqrt(np.mean((preds_years - targets_years) ** 2))
    r = np.corrcoef(preds_years, targets_years)[0, 1]
    
    return {
        'predictions': preds_years,
        'targets': targets_years,
        'mae': mae,
        'rmse': rmse,
        'correlation': r
    }

def ensemble_predictions(cnn_preds, vit_preds, weights=None):
    """Combine CNN and ViT predictions"""
    if weights is None:
        weights = [0.5, 0.5]  # Equal weight
    
    ensemble = weights[0] * cnn_preds + weights[1] * vit_preds
    return ensemble

def plot_comparison(cnn_result, vit_result, ensemble_result, save_path):
    """Plot comparison of models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CNN vs ViT vs Ensemble Comparison', fontsize=16, fontweight='bold')
    
    targets = cnn_result['targets']
    
    # 1. CNN predictions
    axes[0, 0].scatter(targets, cnn_result['predictions'], alpha=0.6, s=40, color='blue')
    axes[0, 0].plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
    axes[0, 0].set_xlabel('Chronological Age (years)')
    axes[0, 0].set_ylabel('Predicted Age (years)')
    axes[0, 0].set_title(f"CNN\nMAE={cnn_result['mae']:.2f}, r={cnn_result['correlation']:.3f}", 
                        fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ViT predictions
    axes[0, 1].scatter(targets, vit_result['predictions'], alpha=0.6, s=40, color='green')
    axes[0, 1].plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
    axes[0, 1].set_xlabel('Chronological Age (years)')
    axes[0, 1].set_ylabel('Predicted Age (years)')
    axes[0, 1].set_title(f"ViT\nMAE={vit_result['mae']:.2f}, r={vit_result['correlation']:.3f}", 
                        fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Ensemble predictions
    axes[0, 2].scatter(targets, ensemble_result['predictions'], alpha=0.6, s=40, color='purple')
    axes[0, 2].plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
    axes[0, 2].set_xlabel('Chronological Age (years)')
    axes[0, 2].set_ylabel('Predicted Age (years)')
    axes[0, 2].set_title(f"Ensemble\nMAE={ensemble_result['mae']:.2f}, r={ensemble_result['correlation']:.3f}", 
                        fontweight='bold', color='purple')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. MAE comparison
    models = ['CNN', 'ViT', 'Ensemble']
    maes = [cnn_result['mae'], vit_result['mae'], ensemble_result['mae']]
    colors = ['blue', 'green', 'purple']
    
    bars = axes[1, 0].bar(models, maes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('MAE (years)', fontsize=12)
    axes[1, 0].set_title('Mean Absolute Error Comparison', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{mae:.2f}',
                       ha='center', va='bottom', fontweight='bold')
    
    # 5. Correlation comparison
    correlations = [cnn_result['correlation'], vit_result['correlation'], ensemble_result['correlation']]
    
    bars = axes[1, 1].bar(models, correlations, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Correlation (r)', fontsize=12)
    axes[1, 1].set_title('Correlation Comparison', fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{corr:.3f}',
                       ha='center', va='bottom', fontweight='bold')
    
    # 6. Residuals comparison
    cnn_residuals = np.abs(cnn_result['predictions'] - cnn_result['targets'])
    vit_residuals = np.abs(vit_result['predictions'] - vit_result['targets'])
    ensemble_residuals = np.abs(ensemble_result['predictions'] - ensemble_result['targets'])
    
    axes[1, 2].boxplot([cnn_residuals, vit_residuals, ensemble_residuals],
                       labels=models,
                       patch_artist=True,
                       widths=0.6)
    
    # Color the boxes
    for patch, color in zip(axes[1, 2].artists, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 2].set_ylabel('Absolute Error (years)', fontsize=12)
    axes[1, 2].set_title('Error Distribution', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Comparison plot saved: {save_path}")
    plt.close()

def main():
    print("=" * 70)
    print(" " * 20 + "MODEL COMPARISON")
    print("=" * 70)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    
    print(f"\n🖥️  Device: {DEVICE}")
    
    # Load validation dataset
    print(f"\n{'='*70}")
    print("Loading Dataset")
    print("=" * 70)
    
    val_dataset, base_dataset = load_dataset()
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\n   Validation samples: {len(val_dataset)}")
    print(f"   Age range: {base_dataset.age_min:.0f} - {base_dataset.age_max:.0f} years")
    
    # Load CNN
    print(f"\n{'='*70}")
    print("Loading CNN Model")
    print("=" * 70)
    
    cnn_model = BrainAgeCNN(pretrained=True, dropout=0.4, num_input_channels=3).to(DEVICE)
    cnn_checkpoint = torch.load('outputs/models/best_cnn.pth', map_location=DEVICE)
    cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
    
    print(f"✅ CNN loaded (Epoch {cnn_checkpoint['epoch']}, MAE: {cnn_checkpoint['val_mae']:.2f})")
    
    # Load ViT
    print(f"\n{'='*70}")
    print("Loading Vision Transformer Model")
    print("=" * 70)
    
    vit_model = BrainAgeViT(
        image_size=224,
        patch_size=16,
        num_channels=3,
        dim=384,
        depth=6,
        heads=6,
        mlp_dim=768,
        dropout=0.2
    ).to(DEVICE)
    
    try:
        vit_checkpoint = torch.load('outputs/models/best_vit.pth', map_location=DEVICE)
        vit_model.load_state_dict(vit_checkpoint['model_state_dict'])
        print(f"✅ ViT loaded (Epoch {vit_checkpoint['epoch']}, MAE: {vit_checkpoint['val_mae']:.2f})")
    except FileNotFoundError:
        print("❌ ViT model not found. Please train ViT first using: python 3_train_vit.py")
        return
    
    # Evaluate both models
    print(f"\n{'='*70}")
    print("Evaluating Models")
    print("=" * 70)
    
    cnn_result = evaluate_model(cnn_model, val_loader, DEVICE, val_dataset, "CNN")
    vit_result = evaluate_model(vit_model, val_loader, DEVICE, val_dataset, "ViT")
    
    # Create ensemble predictions
    ensemble_preds = ensemble_predictions(cnn_result['predictions'], vit_result['predictions'])
    ensemble_targets = cnn_result['targets']
    ensemble_mae = np.mean(np.abs(ensemble_preds - ensemble_targets))
    ensemble_rmse = np.sqrt(np.mean((ensemble_preds - ensemble_targets) ** 2))
    ensemble_r = np.corrcoef(ensemble_preds, ensemble_targets)[0, 1]
    
    ensemble_result = {
        'predictions': ensemble_preds,
        'targets': ensemble_targets,
        'mae': ensemble_mae,
        'rmse': ensemble_rmse,
        'correlation': ensemble_r
    }
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n📊 CNN:")
    print(f"   MAE: {cnn_result['mae']:.2f} years")
    print(f"   RMSE: {cnn_result['rmse']:.2f} years")
    print(f"   Correlation: {cnn_result['correlation']:.3f}")
    
    print(f"\n📊 Vision Transformer:")
    print(f"   MAE: {vit_result['mae']:.2f} years")
    print(f"   RMSE: {vit_result['rmse']:.2f} years")
    print(f"   Correlation: {vit_result['correlation']:.3f}")
    
    print(f"\n📊 Ensemble (50% CNN + 50% ViT):")
    print(f"   MAE: {ensemble_result['mae']:.2f} years")
    print(f"   RMSE: {ensemble_result['rmse']:.2f} years")
    print(f"   Correlation: {ensemble_result['correlation']:.3f}")
    
    # Calculate improvements
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("=" * 70)
    
    vit_improvement = cnn_result['mae'] - vit_result['mae']
    ensemble_improvement = cnn_result['mae'] - ensemble_result['mae']
    
    print(f"\n🔍 ViT vs CNN:")
    if vit_improvement > 0:
        print(f"   ✅ ViT improved by {vit_improvement:.2f} years ({vit_improvement/cnn_result['mae']*100:.1f}%)")
    else:
        print(f"   ⚠️  CNN was {-vit_improvement:.2f} years better")
    
    print(f"\n🔍 Ensemble vs CNN:")
    if ensemble_improvement > 0:
        print(f"   ✅ Ensemble improved by {ensemble_improvement:.2f} years ({ensemble_improvement/cnn_result['mae']*100:.1f}%)")
    else:
        print(f"   ⚠️  CNN was {-ensemble_improvement:.2f} years better")
    
    best_model = min(
        [('CNN', cnn_result['mae']), ('ViT', vit_result['mae']), ('Ensemble', ensemble_result['mae'])],
        key=lambda x: x[1]
    )
    print(f"\n🏆 Best Model: {best_model[0]} (MAE: {best_model[1]:.2f} years)")
    
    # Save comparison results
    Path('outputs').mkdir(exist_ok=True)
    
    comparison_data = {
        'cnn': {
            'mae': float(cnn_result['mae']),
            'rmse': float(cnn_result['rmse']),
            'correlation': float(cnn_result['correlation']),
            'epoch': int(cnn_checkpoint['epoch'])
        },
        'vit': {
            'mae': float(vit_result['mae']),
            'rmse': float(vit_result['rmse']),
            'correlation': float(vit_result['correlation']),
            'epoch': int(vit_checkpoint['epoch'])
        },
        'ensemble': {
            'mae': float(ensemble_result['mae']),
            'rmse': float(ensemble_result['rmse']),
            'correlation': float(ensemble_result['correlation']),
            'weights': [0.5, 0.5]
        }
    }
    
    with open('outputs/model_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Plot comparison
    plot_comparison(cnn_result, vit_result, ensemble_result, 'outputs/model_comparison.png')
    
    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\n📂 Outputs:")
    print(f"   Comparison: outputs/model_comparison.json")
    print(f"   Plot: outputs/model_comparison.png")

if __name__ == "__main__":
    main()