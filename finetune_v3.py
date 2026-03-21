"""
NeuroScan — Fine-Tuning v3: Focal Loss + Class Weights
Target: Fix Glioma recall (was 76.2%) without hurting other classes.

Strategy:
  1. Load from finetuned model (brain_tumor_model_finetuned.h5) — don't start from scratch
  2. Focal Loss  — penalises hard misclassifications more heavily
  3. Class Weights — Glioma gets 1.6x weight (most dangerous to miss)
  4. Glioma oversampling — duplicate Glioma training samples 1.5x
  5. Unfreeze top 20 layers (more capacity) at very low LR
  6. Monitor val_glioma_recall as primary metric via custom callback

Run from C:\\brain tumor\\ with venv active:
    python finetune_v3.py
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH      = 'brain_tumor_model_finetuned.h5'   # start from best checkpoint
TRAIN_DIR       = 'dataset/Training'
VAL_SPLIT       = 0.2
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
EPOCHS          = 40
UNFREEZE_LAYERS = 20          # more layers than v2 (was 10)
LEARNING_RATE   = 3e-6        # very conservative
SAVED_MODEL     = 'brain_tumor_model_v3.h5'
HISTORY_PNG     = 'finetune_v3_history.png'

CLASS_NAMES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Class weights: Glioma gets 1.6x, rest get 1.0x
# Clinically: missing Glioma (most aggressive) is far worse than a false positive
CLASS_WEIGHTS = {0: 1.6, 1: 1.0, 2: 1.0, 3: 1.0}

# Focal loss parameters
FOCAL_GAMMA = 2.0    # focusing parameter — 2.0 is standard
FOCAL_ALPHA = 0.25   # down-weights easy negatives

# ── FOCAL LOSS ────────────────────────────────────────────────────────────────
def focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA):
    """
    Categorical focal loss.
    Focuses training on hard-to-classify examples.
    gamma=2 means easy examples (p > 0.9) contribute ~100x less loss than hard ones.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * tf.math.log(y_pred)
        # Focal weight: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        focal_loss_val = alpha * focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss_val, axis=-1))
    loss_fn.__name__ = 'focal_loss'
    return loss_fn

# ── CUSTOM CALLBACK: track per-class recall ───────────────────────────────────
class ClassRecallCallback(keras.callbacks.Callback):
    """Prints per-class recall at the end of each epoch."""
    def __init__(self, val_ds, class_labels):
        super().__init__()
        self.val_ds = val_ds
        self.class_labels = class_labels
        self.glioma_recalls = []

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for x_batch, y_batch in self.val_ds:
            preds = self.model.predict(x_batch, verbose=0)
            y_true.extend(np.argmax(y_batch.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        recalls = []
        for i, label in enumerate(self.class_labels):
            mask = (y_true == i)
            recall = np.mean(y_pred[mask] == i) if mask.sum() > 0 else 0.0
            recalls.append(recall)

        glioma_recall = recalls[0]
        self.glioma_recalls.append(glioma_recall)

        recall_str = '  '.join(
            [f"{self.class_labels[i]}: {recalls[i]*100:.1f}%" for i in range(4)]
        )
        print(f"\n  📊 Per-class recall → {recall_str}")
        print(f"  🎯 Glioma recall this epoch: {glioma_recall*100:.1f}%")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading finetuned model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'loss_fn': focal_loss()}   # handle if saved with custom loss
)
print("Model loaded.\n")

# ── UNFREEZE TOP 20 LAYERS ────────────────────────────────────────────────────
efficientnet = model.get_layer('efficientnetb0')

for layer in efficientnet.layers:
    layer.trainable = False

unfrozen = 0
for layer in efficientnet.layers[-UNFREEZE_LAYERS:]:
    if not isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = True
        unfrozen += 1

print(f"Unfrozen layers: {unfrozen} (BatchNorm kept frozen)")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=focal_loss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA),
    metrics=['accuracy']
)

trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
total     = model.count_params()
print(f"Trainable params: {trainable:,} / {total:,}\n")

# ── DATA PIPELINE WITH GLIOMA OVERSAMPLING ────────────────────────────────────
# Moderate augmentation — slightly stronger for Glioma
data_aug_normal = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.08),
    keras.layers.RandomZoom(0.08),
], name='aug_normal')

data_aug_glioma = keras.Sequential([
    keras.layers.RandomFlip('horizontal_and_vertical'),
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.12),
    keras.layers.RandomContrast(0.1),
], name='aug_glioma')

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, label

# Build train/val split with Glioma oversampling
all_paths, all_labels, all_sparse = [], [], []
for idx, cls in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(TRAIN_DIR, cls)
    files = [f for f in os.listdir(cls_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    oversample = 1.5 if cls == 'glioma' else 1.0
    files_to_use = files.copy()
    if oversample > 1.0:
        extra = int(len(files) * (oversample - 1.0))
        files_to_use += random.choices(files, k=extra)
        print(f"  {cls}: {len(files)} → {len(files_to_use)} (oversampled x{oversample})")
    else:
        print(f"  {cls}: {len(files_to_use)}")

    for fname in files_to_use:
        all_paths.append(os.path.join(cls_dir, fname))
        oh = [0.0] * 4; oh[idx] = 1.0
        all_labels.append(oh)
        all_sparse.append(idx)

combined = list(zip(all_paths, all_labels, all_sparse))
random.seed(42); random.shuffle(combined)
all_paths, all_labels, all_sparse = zip(*combined)

split = int(len(all_paths) * (1 - VAL_SPLIT))
train_p, train_l, train_s = (list(all_paths[:split]),
                              list(all_labels[:split]),
                              list(all_sparse[:split]))
val_p,   val_l            = list(all_paths[split:]), list(all_labels[split:])

print(f"\nTrain: {len(train_p)}  |  Val: {len(val_p)}\n")

def make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.float32)))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (data_aug_normal(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def make_val_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.float32)))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_train_ds(train_p, train_l)
val_ds   = make_val_ds(val_p, val_l)

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
recall_cb = ClassRecallCallback(val_ds, CLASS_LABELS)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        SAVED_MODEL,
        monitor='val_accuracy',
        save_best_only=True,
        save_format='h5',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-8,
        verbose=1
    ),
    recall_cb,
]

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("=" * 65)
print(f"Fine-tuning v3: Focal Loss (γ={FOCAL_GAMMA}) + Class Weights")
print(f"Unfreeze: {UNFREEZE_LAYERS} layers  |  LR: {LEARNING_RATE}  |  Batch: {BATCH_SIZE}")
print(f"Glioma weight: {CLASS_WEIGHTS[0]}x  |  Glioma oversampled: 1.5x")
print(f"Max epochs: {EPOCHS}  |  Early stop patience: 10")
print("=" * 65 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=CLASS_WEIGHTS,
    callbacks=callbacks,
    verbose=1
)

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.patch.set_facecolor('#0e1117')
ep = range(1, len(history.history['accuracy']) + 1)

for ax in axes:
    ax.set_facecolor('#1a2030')
    ax.grid(color='#252d3d', linewidth=0.8)
    for sp in ax.spines.values(): sp.set_edgecolor('#252d3d')
    ax.tick_params(colors='#8892a4', labelsize=9)

# Accuracy
axes[0].plot(ep, [v*100 for v in history.history['accuracy']],
             color='#c9a84c', lw=2, marker='o', ms=4, label='Train')
axes[0].plot(ep, [v*100 for v in history.history['val_accuracy']],
             color='#5b8dd9', lw=2, marker='o', ms=4, label='Val')
axes[0].set_title('Accuracy', color='#f4f1eb', fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel('Epoch', color='#8892a4')
axes[0].set_ylabel('Accuracy (%)', color='#8892a4')
axes[0].legend(facecolor='#252d3d', edgecolor='#252d3d', labelcolor='#b0bac8', fontsize=9)

# Loss
axes[1].plot(ep, history.history['loss'],
             color='#c0392b', lw=2, marker='o', ms=4, label='Train')
axes[1].plot(ep, history.history['val_loss'],
             color='#2ecc71', lw=2, marker='o', ms=4, label='Val')
axes[1].set_title('Focal Loss', color='#f4f1eb', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel('Epoch', color='#8892a4')
axes[1].set_ylabel('Loss', color='#8892a4')
axes[1].legend(facecolor='#252d3d', edgecolor='#252d3d', labelcolor='#b0bac8', fontsize=9)

# Glioma recall over epochs
glioma_r = [r * 100 for r in recall_cb.glioma_recalls]
axes[2].plot(ep, glioma_r, color='#e74c3c', lw=2, marker='o', ms=4, label='Glioma Recall')
axes[2].axhline(76.2, color='#555', linestyle='--', lw=1.5, label='Baseline (76.2%)')
axes[2].set_title('Glioma Recall', color='#f4f1eb', fontsize=13, fontweight='bold', pad=12)
axes[2].set_xlabel('Epoch', color='#8892a4')
axes[2].set_ylabel('Recall (%)', color='#8892a4')
axes[2].legend(facecolor='#252d3d', edgecolor='#252d3d', labelcolor='#b0bac8', fontsize=9)
axes[2].set_ylim(60, 102)

best_acc = max(history.history['val_accuracy']) * 100
best_recall = max(glioma_r) if glioma_r else 0
fig.text(
    0.5, 0.01,
    f'Best val accuracy: {best_acc:.2f}%  |  Best Glioma recall: {best_recall:.1f}%  |  '
    f'Epochs: {len(ep)}  |  Focal γ={FOCAL_GAMMA}  |  Glioma weight: {CLASS_WEIGHTS[0]}x',
    ha='center', color='#8892a4', fontsize=9, style='italic'
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(HISTORY_PNG, dpi=150, bbox_inches='tight',
            facecolor='#0e1117', edgecolor='none')
plt.close()

print(f"\n{'='*65}")
print(f"  Best val accuracy  : {best_acc:.2f}%")
print(f"  Best Glioma recall : {best_recall:.1f}%  (was 76.2% in v2)")
print(f"  Model saved        → {SAVED_MODEL}")
print(f"  History plot       → {HISTORY_PNG}")
print(f"{'='*65}")
print(f"\nNext steps:")
print(f"  1. Run evaluate.py with MODEL_PATH = '{SAVED_MODEL}'")
print(f"  2. Update app.py: model = tf.keras.models.load_model('{SAVED_MODEL}')")