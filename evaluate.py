"""
NeuroScan — Model Evaluation Script
Generates confusion matrix + per-class metrics on the test set.
Run from C:\\brain tumor\\ with venv active:
    python evaluate.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import seaborn as sns

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = 'brain_tumor_model_v3.h5'
TEST_DIR     = 'dataset/Testing'
IMG_SIZE     = (224, 224)
OUTPUT_PNG   = 'confusion_matrix.png'
OUTPUT_TXT   = 'metrics_report.txt'

# Must match the order your model was trained on
CLASS_NAMES  = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
FOLDER_MAP   = {          # folder name → class index
    'glioma':      0,
    'meningioma':  1,
    'notumor':     2,
    'pituitary':   3,
}
COLORS = ['#c0392b', '#b8620a', '#1e7e5a', '#c9a84c']

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.\n")

# ── LOAD TEST IMAGES ──────────────────────────────────────────────────────────
print("Loading test images...")
y_true, y_pred = [], []
total = 0

for folder_name, class_idx in FOLDER_MAP.items():
    folder_path = os.path.join(TEST_DIR, folder_name)
    if not os.path.exists(folder_path):
        print(f"  WARNING: folder not found: {folder_path}")
        continue

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"  {CLASS_NAMES[class_idx]}: {len(files)} images")

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        try:
            img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32)
            arr = preprocess_input(arr)
            arr = np.expand_dims(arr, 0)

            preds = model.predict(arr, verbose=0)[0]
            predicted = int(np.argmax(preds))

            y_true.append(class_idx)
            y_pred.append(predicted)
            total += 1
        except Exception as e:
            print(f"    Skipped {fname}: {e}")

print(f"\nTotal images evaluated: {total}\n")

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ── METRICS ───────────────────────────────────────────────────────────────────
overall_acc = np.mean(y_true == y_pred) * 100

precision = precision_score(y_true, y_pred, average=None)
recall    = recall_score(y_true, y_pred, average=None)
f1        = f1_score(y_true, y_pred, average=None)

macro_f1  = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

# ── PRINT TO CONSOLE ──────────────────────────────────────────────────────────
print("=" * 60)
print(f"  OVERALL TEST ACCURACY : {overall_acc:.2f}%")
print(f"  MACRO F1 SCORE        : {macro_f1:.4f}")
print(f"  WEIGHTED F1 SCORE     : {weighted_f1:.4f}")
print("=" * 60)
print("\nPer-Class Metrics:")
print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 48)
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f}")
print("\nFull Classification Report:")
print(report)

# ── SAVE TEXT REPORT ──────────────────────────────────────────────────────────
with open(OUTPUT_TXT, 'w') as f:
    f.write("NeuroScan — Model Evaluation Report\n")
    f.write("=" * 60 + "\n")
    f.write(f"Overall Test Accuracy : {overall_acc:.2f}%\n")
    f.write(f"Macro F1 Score        : {macro_f1:.4f}\n")
    f.write(f"Weighted F1 Score     : {weighted_f1:.4f}\n")
    f.write("=" * 60 + "\n\n")
    f.write("Per-Class Metrics:\n")
    f.write(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
    f.write("-" * 48 + "\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"{name:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f}\n")
    f.write("\nFull Classification Report:\n")
    f.write(report)
print(f"Text report saved → {OUTPUT_TXT}")

# ── CONFUSION MATRIX PLOT ─────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100  # row-normalised %

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#0e1117')

# ── LEFT: Confusion matrix heatmap ───────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor('#0e1117')

# Custom colormap: dark background → gold
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    'neuroscan', ['#1a2030', '#c9a84c'], N=256
)

sns.heatmap(
    cm_pct, annot=False, fmt='.1f', cmap=cmap,
    linewidths=0.5, linecolor='#252d3d',
    ax=ax1, cbar=True,
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    vmin=0, vmax=100
)

# Annotate each cell with count + percentage
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        count = cm[i, j]
        pct   = cm_pct[i, j]
        color = 'white' if pct < 55 else '#0e1117'
        ax1.text(j + 0.5, i + 0.42,
                 f'{count}',
                 ha='center', va='center',
                 fontsize=13, fontweight='bold', color=color)
        ax1.text(j + 0.5, i + 0.62,
                 f'{pct:.1f}%',
                 ha='center', va='center',
                 fontsize=9, color=color, alpha=0.85)

ax1.set_title('Confusion Matrix', color='#f4f1eb',
              fontsize=14, fontweight='bold', pad=16)
ax1.set_xlabel('Predicted', color='#8892a4', fontsize=11, labelpad=10)
ax1.set_ylabel('Actual', color='#8892a4', fontsize=11, labelpad=10)
ax1.tick_params(colors='#b0bac8', labelsize=10)
for spine in ax1.spines.values():
    spine.set_edgecolor('#252d3d')

cbar = ax1.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color='#8892a4')
cbar.ax.tick_params(labelsize=9, colors='#8892a4')
cbar.set_label('Row %', color='#8892a4', fontsize=9)

# ── RIGHT: Per-class bar chart ────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#1a2030')

metrics = {'Precision': precision, 'Recall': recall, 'F1 Score': f1}
x = np.arange(len(CLASS_NAMES))
width = 0.24
offsets = [-width, 0, width]
metric_colors = ['#5b8dd9', '#c9a84c', '#2ecc71']

for idx, (metric_name, values) in enumerate(metrics.items()):
    bars = ax2.bar(
        x + offsets[idx], values * 100, width,
        color=metric_colors[idx], alpha=0.85,
        label=metric_name, zorder=3
    )
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f'{val*100:.1f}',
            ha='center', va='bottom',
            fontsize=8, color='#b0bac8'
        )

ax2.set_xticks(x)
ax2.set_xticklabels(CLASS_NAMES, color='#b0bac8', fontsize=10)
ax2.set_ylim(0, 112)
ax2.set_ylabel('Score (%)', color='#8892a4', fontsize=11)
ax2.set_title('Per-Class Metrics', color='#f4f1eb',
              fontsize=14, fontweight='bold', pad=16)
ax2.tick_params(axis='y', colors='#8892a4', labelsize=9)
ax2.tick_params(axis='x', colors='#b0bac8')
ax2.set_facecolor('#1a2030')
ax2.grid(axis='y', color='#252d3d', linewidth=0.8, zorder=0)
for spine in ax2.spines.values():
    spine.set_edgecolor('#252d3d')
ax2.legend(
    facecolor='#252d3d', edgecolor='#252d3d',
    labelcolor='#b0bac8', fontsize=9,
    loc='lower right'
)

# ── Overall stats text ────────────────────────────────────────────────────────
fig.text(
    0.5, 0.02,
    f'Test Accuracy: {overall_acc:.2f}%   |   '
    f'Macro F1: {macro_f1*100:.2f}%   |   '
    f'Weighted F1: {weighted_f1*100:.2f}%   |   '
    f'Total samples: {total}',
    ha='center', color='#8892a4', fontsize=10,
    style='italic'
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight',
            facecolor='#0e1117', edgecolor='none')
plt.close()
print(f"Confusion matrix saved → {OUTPUT_PNG}")
print("\nDone! Check confusion_matrix.png and metrics_report.txt")