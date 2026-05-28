"""
Resume or Check Training Status
If you interrupted training with Ctrl+C, use this script to:
1. Check if a best model was saved
2. Load and evaluate the best model
3. Resume training from checkpoint (if desired)
Run: python check_training_status.py
"""

import torch
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset

from utils.dataset import BrainAgeDataset
from models.cnn_model import BrainAgeCNN

def denormalize_ages(ages_normalized, dataset):
    """Convert normalized ages back to years"""
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    ages_years = np.array(ages_normalized) * base_dataset.age_std + base_dataset.age_mean
    return ages_years

def main():
    print("=" * 70)
    print("CHECKING TRAINING STATUS")
    print("=" * 70)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 16
    
    # Check if best model exists
    checkpoint_path = Path('outputs/models/best_cnn.pth')
    
    if not checkpoint_path.exists():
        print("\n❌ No checkpoint found!")
        print("   Location: outputs/models/best_cnn.pth")
        print("\n⚠️  If you terminated training early:")
        print("   1. Check if data/processed/ exists")
        print("   2. Run: python 1_data_preparation.py (if needed)")
        print("   3. Start fresh: python 2_train_cnn.py")
        return
    
    print("\n✅ Best model checkpoint found!")
    
    # Load checkpoint
    print("\n" + "=" * 70)
    print("Loading Checkpoint")
    print("=" * 70)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    print(f"\n📊 Checkpoint Info:")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Validation MAE: {checkpoint['val_mae']:.2f} years")
    print(f"   Validation Correlation: {checkpoint['val_r']:.3f}")
    
    # Load model
    model = BrainAgeCNN(pretrained=True, dropout=0.4, num_input_channels=3).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load training history
    history = checkpoint['history']
    
    print(f"\n📈 Training History:")
    print(f"   Total epochs completed: {len(history['train_mae'])}")
    print(f"   Best epoch: {np.argmin(history['val_mae']) + 1}")
    print(f"   Final train MAE: {history['train_mae'][-1]:.2f} years")
    print(f"   Final val MAE: {history['val_mae'][-1]:.2f} years")
    
    # Load dataset for evaluation
    print(f"\n{'='*70}")
    print("Loading Dataset for Evaluation")
    print("=" * 70)
    
    val_dataset_full = BrainAgeDataset('data/processed', augment=False)
    
    # Use same split as training
    np.random.seed(42)
    indices = np.random.permutation(len(val_dataset_full))
    train_size = int(0.8 * len(indices))
    
    val_indices = indices[train_size:]
    val_dataset = Subset(val_dataset_full, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\n   Validation samples: {len(val_dataset)}")
    
    # Evaluate on validation set
    print(f"\n{'='*70}")
    print("Evaluating Model on Validation Set")
    print("=" * 70)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, ages_norm, _ in val_loader:
            images = images.to(DEVICE)
            ages_norm = ages_norm.to(DEVICE)
            
            preds_norm = model(images)
            
            all_preds.extend(preds_norm.cpu().numpy())
            all_targets.extend(ages_norm.cpu().numpy())
    
    # Denormalize
    preds_years = denormalize_ages(all_preds, val_dataset)
    targets_years = denormalize_ages(all_targets, val_dataset)
    
    mae = np.mean(np.abs(preds_years - targets_years))
    rmse = np.sqrt(np.mean((preds_years - targets_years) ** 2))
    r = np.corrcoef(preds_years, targets_years)[0, 1]
    residuals = preds_years - targets_years
    
    print(f"\n🏆 Current Model Performance:")
    print(f"   MAE: {mae:.2f} years")
    print(f"   RMSE: {rmse:.2f} years")
    print(f"   Correlation: {r:.3f}")
    print(f"   Mean Error (bias): {np.mean(residuals):.2f} years")
    print(f"   Std of Error: {np.std(residuals):.2f} years")
    print(f"   Median Error: {np.median(np.abs(residuals)):.2f} years")
    
    # Age group analysis
    print(f"\n📊 Performance by Age Group:")
    age_groups = [(18, 40), (40, 60), (60, 80), (80, 100)]
    for age_min, age_max in age_groups:
        mask = (targets_years >= age_min) & (targets_years < age_max)
        if mask.sum() > 0:
            group_mae = np.mean(np.abs(preds_years[mask] - targets_years[mask]))
            group_r = np.corrcoef(preds_years[mask], targets_years[mask])[0, 1]
            print(f"   Age {age_min}-{age_max}: MAE={group_mae:.2f}, r={group_r:.3f}, n={mask.sum()}")
    
    # Status and recommendations
    print(f"\n{'='*70}")
    print("STATUS & RECOMMENDATIONS")
    print("=" * 70)
    
    if mae < 7:
        print(f"\n✅ GOOD: Model performance is excellent (MAE: {mae:.2f} years)")
        print("\n   Next steps:")
        print("   1. ✓ CNN training is complete")
        print("   2. → Train Vision Transformer: python 3_train_vit.py")
        print("   3. → Compare models: python compare_models.py")
        print("   4. → Run inference: python 4_inference.py")
    elif mae < 10:
        print(f"\n⚠️  ACCEPTABLE: Model performance is reasonable (MAE: {mae:.2f} years)")
        print("\n   Consider:")
        print("   1. Continue training to potentially improve")
        print("   2. Or proceed to Vision Transformer for comparison")
        print("   3. Or create ensemble if both models are trained")
    else:
        print(f"\n❌ NEEDS IMPROVEMENT: MAE is high ({mae:.2f} years)")
        print("\n   Recommendations:")
        print("   1. Check if training was interrupted very early")
        print("   2. Verify data preparation (python 1_data_preparation.py)")
        print("   3. Restart training (python 2_train_cnn.py)")
    
    # Check if training history file exists
    history_file = Path('outputs/training_history.json')
    if history_file.exists():
        print(f"\n📂 Training history saved: outputs/training_history.json")
    
    # Offer to run next step
    print(f"\n{'='*70}")
    print("WHAT TO DO NEXT")
    print("=" * 70)
    
    choice = input(
        "\nYour options:\n"
        "1. Train Vision Transformer (python 3_train_vit.py)\n"
        "2. Compare CNN vs ViT (python compare_models.py)\n"
        "3. Run inference (python 4_inference.py)\n"
        "4. Resume CNN training from this checkpoint\n"
        "5. Exit\n"
        "\nChoice (1-5): "
    ).strip()
    
    if choice == '1':
        print("\n→ To train Vision Transformer, run:")
        print("   python 3_train_vit.py")
    elif choice == '2':
        print("\n⚠️  Vision Transformer must be trained first!")
        print("   Run: python 3_train_vit.py")
    elif choice == '3':
        print("\n→ To run inference, run:")
        print("   python 4_inference.py")
    elif choice == '4':
        print("\n→ To resume CNN training from checkpoint, run:")
        print("   python 2_train_cnn_resume.py")
        print("\n   (Note: You can modify 2_train_cnn.py to resume from checkpoint)")
    else:
        print("\n✓ Exiting")
        return
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()