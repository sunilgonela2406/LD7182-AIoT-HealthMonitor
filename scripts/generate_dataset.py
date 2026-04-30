"""
LD7182 AIoT Health Monitor — Synthetic Dataset Generator
Author: [YOUR NAME]
Description: Generates labelled health monitoring dataset for Edge Impulse training
             based on WHO/BNF clinical threshold guidelines.

Usage: python generate_dataset.py
Output: health_dataset.csv (upload to Edge Impulse Data Acquisition)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N_PER_CLASS = 800  # samples per class (total: 2400)


def add_noise(arr, sigma):
    """Add Gaussian noise to simulate sensor measurement variability."""
    return arr + np.random.normal(0, sigma, len(arr))


# ── Class 0: NORMAL ──────────────────────────────────────────────
# SpO2: 95-100%, HR: 60-100 bpm, Temp: 36.1-37.5°C
spo2_n  = add_noise(np.random.uniform(95, 100,   N_PER_CLASS), 0.5)
hr_n    = add_noise(np.random.uniform(60,  100,  N_PER_CLASS), 2.0)
temp_n  = add_noise(np.random.uniform(36.1, 37.5, N_PER_CLASS), 0.1)
labels_n = np.zeros(N_PER_CLASS, dtype=int)

# ── Class 1: CAUTION ─────────────────────────────────────────────
# SpO2: 92-94%, HR: 101-120 bpm
spo2_c  = add_noise(np.random.uniform(92, 94,   N_PER_CLASS), 0.3)
hr_c    = add_noise(np.random.uniform(101, 120, N_PER_CLASS), 2.0)
temp_c  = add_noise(np.random.uniform(36.1, 38.0, N_PER_CLASS), 0.2)
labels_c = np.ones(N_PER_CLASS, dtype=int)

# ── Class 2: ALERT ───────────────────────────────────────────────
# SpO2: < 92%, HR: > 120 bpm, Temp: > 38.5°C
spo2_a  = add_noise(np.random.uniform(85, 92,   N_PER_CLASS), 0.5)
hr_a    = add_noise(np.random.uniform(120, 160, N_PER_CLASS), 3.0)
temp_a  = add_noise(np.random.uniform(38.5, 40.5, N_PER_CLASS), 0.2)
labels_a = np.full(N_PER_CLASS, 2, dtype=int)

# ── Combine, clip, and shuffle ────────────────────────────────────
spo2   = np.clip(np.concatenate([spo2_n,  spo2_c,  spo2_a]),  80, 100)
hr     = np.clip(np.concatenate([hr_n,    hr_c,    hr_a]),     40, 200)
temp   = np.clip(np.concatenate([temp_n,  temp_c,  temp_a]),   35, 42)
labels = np.concatenate([labels_n, labels_c, labels_a])

idx = np.random.permutation(len(labels))
df = pd.DataFrame({
    'spo2':        spo2[idx].round(2),
    'heart_rate':  hr[idx].round(1),
    'temperature': temp[idx].round(2),
    'label':       labels[idx]
})

df.to_csv('health_dataset.csv', index=False)
print(f"Dataset generated: {len(df)} samples")
print("\nClass distribution:")
label_names = {0: 'NORMAL', 1: 'CAUTION', 2: 'ALERT'}
for cls, name in label_names.items():
    count = (df['label'] == cls).sum()
    print(f"  {name}: {count} samples ({count/len(df)*100:.1f}%)")
print("\nFeature statistics:")
print(df[['spo2', 'heart_rate', 'temperature']].describe().round(2))
print("\nSaved: health_dataset.csv")
print("Next step: Upload to Edge Impulse > Data Acquisition > Upload existing data")
