"""
Brain Age Inference with GradCAM
Run: python 4_inference.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from models.cnn_model import BrainAgeCNN
from utils.gradcam import GradCAM, overlay_heatmap
from utils.dataset import BrainAgeDataset

def generate_report(subject_id, chrono_age, pred_age, gap):
    """Generate clinical report"""
    
    interpretation = "Normal aging"
    if gap > 5:
        interpretation = "Brain appears older (possible neurodegeneration)"
    elif gap < -5:
        interpretation = "Brain appears younger (preserved health)"
    
    report = {
        'subject_id': subject_id,
        'chronological_age': float(chrono_age),
        'predicted_age': float(pred_age),
        'brain_age_gap': float(gap),
        'interpretation': interpretation,
        'confidence_interval': '±4.5 years'
    }
    
    return report

def run_inference(subject_idx=0):
    """Run inference on a subject"""
    
    print("=" * 70)
    print("BRAIN AGE INFERENCE WITH GRADCAM")
    print("=" * 70)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = BrainAgeDataset('data/processed', augment=False)
    image, age_norm, subject_id = dataset[subject_idx]
    
    # Denormalize age
    chrono_age = dataset.denormalize_age(age_norm.item())
    
    print(f"\n📋 Subject: {subject_id}")
    print(f"📅 Chronological Age: {chrono_age:.1f} years")
    
    # Load model
    model = BrainAgeCNN(pretrained=True, dropout=0.4, num_input_channels=3).to(DEVICE)
    checkpoint = torch.load('outputs/models/best_cnn.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Predict
    input_tensor = image.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_norm = model(input_tensor).item()
    
    pred_age = dataset.denormalize_age(pred_norm)
    gap = pred_age - chrono_age
    
    print(f"🧠 Predicted Age: {pred_age:.1f} years")
    print(f"📊 Brain Age Gap: {gap:+.1f} years")
    
    # Generate GradCAM
    print(f"\n🔥 Generating GradCAM...")
    
    gradcam = GradCAM(model, model.resnet.layer4)
    input_tensor.requires_grad = True
    heatmap, _ = gradcam.generate_cam(input_tensor)
    gradcam.remove_hooks()
    
    # Generate report
    report = generate_report(subject_id, chrono_age, pred_age, gap)
    
    report_path = Path('outputs/reports')
    report_path.mkdir(parents=True, exist_ok=True)
    
    with open(report_path / f'{subject_id}_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Report saved: outputs/reports/{subject_id}_report.json")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(image[1].cpu().numpy(), cmap='gray')
    axes[0].set_title(f'MRI Slice\nAge: {chrono_age:.0f} years')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('GradCAM Heatmap\n(Aging Regions)')
    axes[1].axis('off')
    
    # Overlay
    overlay = overlay_heatmap(image[1].cpu().numpy(), heatmap)
    axes[2].imshow(overlay)
    axes[2].set_title(f'Predicted: {pred_age:.1f} years\nGap: {gap:+.1f} years')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    viz_path = Path('outputs/visualizations')
    viz_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_path / f'{subject_id}_gradcam.png', dpi=150, bbox_inches='tight')
    
    print(f"📊 Visualization saved: outputs/visualizations/{subject_id}_gradcam.png")
    
    plt.show()
    
    print(f"\n{'='*70}")
    print("✅ INFERENCE COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    run_inference(subject_idx=0)