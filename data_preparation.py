"""
Data Preparation: Download OASIS VBM and extract slices
Run: python 1_data_preparation.py
Time: 20-30 minutes
Storage: ~1.6GB download → ~600MB processed
"""

from nilearn import datasets
import nibabel as nib
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from scipy.ndimage import zoom


def download_and_prepare_data(n_subjects=416, output_dir='data/processed'):
    """
    Download OASIS VBM dataset and prepare slices

    Args:
        n_subjects: Number of subjects (max 416 for OASIS VBM)
        output_dir: Output directory for processed data
    """

    print("=" * 70)
    print("BRAIN AGE PREDICTOR - DATA PREPARATION")
    print("=" * 70)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download OASIS VBM
    print(f"\n📥 Step 1: Downloading OASIS VBM ({n_subjects} subjects)")
    print("First run downloads ~1.6GB (cached for future runs)")
    print("-" * 70)

    oasis = datasets.fetch_oasis_vbm(n_subjects=n_subjects)

    # FIX: Ages are stored in ext_vars
    ages = oasis.ext_vars["age"].values

    print(f"\n✅ Downloaded {len(oasis.gray_matter_maps)} subjects")
    print(f"   Age range: {min(ages):.1f} - {max(ages):.1f} years")
    print(f"   Mean age: {np.mean(ages):.1f} ± {np.std(ages):.1f} years")

    # Process each subject
    print(f"\n🔄 Step 2: Processing 3D volumes → 2D slices")
    print("-" * 70)

    metadata = []
    failed = []

    for idx, (img_path, age) in enumerate(tqdm(
        zip(oasis.gray_matter_maps, ages),
        total=len(ages),
        desc="Processing"
    )):

        try:
            # Load 3D MRI volume
            img = nib.load(img_path)
            data = img.get_fdata()

            # Extract middle 40 axial slices
            z_center = data.shape[2] // 2
            slices = data[:, :, z_center-20:z_center+20]

            # Transpose to (40, H, W)
            slices = np.transpose(slices, (2, 0, 1))

            # Resize to 224x224
            slices_resized = np.zeros((40, 224, 224), dtype=np.float32)

            for s in range(40):
                zoom_factors = (
                    224 / slices.shape[1],
                    224 / slices.shape[2]
                )

                slices_resized[s] = zoom(
                    slices[s],
                    zoom_factors,
                    order=1
                )

            # Standardize (mean=0, std=1)
            mean = slices_resized.mean()
            std = slices_resized.std()

            if std > 1e-6:
                slices_resized = (slices_resized - mean) / std
            else:
                slices_resized = slices_resized - mean

            # Clip outliers
            slices_resized = np.clip(slices_resized, -5, 5)

            # Save compressed
            subject_id = f"subject_{idx:04d}"

            np.savez_compressed(
                f"{output_dir}/{subject_id}.npz",
                slices=slices_resized.astype(np.float16),
                age=float(age)
            )

            metadata.append({
                'subject_id': subject_id,
                'age': float(age),
                'original_path': str(img_path)
            })

        except Exception as e:
            failed.append((idx, str(e)))
            continue

    # Save metadata
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("✅ DATA PREPARATION COMPLETE")
    print("=" * 70)

    print(f"\n📊 Summary:")
    print(f"   ✓ Processed: {len(metadata)} subjects")
    print(f"   ✗ Failed: {len(failed)} subjects")
    print(f"   💾 Storage: ~{len(metadata) * 1.5:.0f} MB")
    print(f"   📂 Location: {output_dir}/")

    # Age distribution
    ages = [m['age'] for m in metadata]

    print(f"\n📈 Age Statistics:")
    print(f"   Range: {min(ages):.0f} - {max(ages):.0f} years")
    print(f"   Mean: {np.mean(ages):.1f} years")
    print(f"   Median: {np.median(ages):.1f} years")
    print(f"   Std: {np.std(ages):.1f} years")

    # Age bins
    bins = [18, 30, 40, 50, 60, 70, 80, 100]
    hist, _ = np.histogram(ages, bins=bins)

    print(f"\n📊 Age Distribution:")

    for i in range(len(bins)-1):
        pct = hist[i] / len(ages) * 100
        bar = "█" * int(pct / 2)

        print(
            f"   {bins[i]:2d}-{bins[i+1]:2d} years: "
            f"{hist[i]:3d} ({pct:4.1f}%) {bar}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    download_and_prepare_data(n_subjects=416)