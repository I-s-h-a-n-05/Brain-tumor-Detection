# train.py — Clean training on new balanced dataset
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from model import build_cnn_model

DATASET_PATH = './dataset'
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    shear_range=0.1,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Training'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Training'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Classes:", train_gen.class_indices)
print(f"Training on {train_gen.samples} images, validating on {val_gen.samples} images\n")

model = build_cnn_model()

callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-7, monitor='val_loss', verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

model.save('brain_tumor_model.keras')
best = max(history.history['val_accuracy'])
print(f"\n✅ Done! Best val accuracy: {best:.4f} ({best*100:.1f}%)")

# Final evaluation on Testing folder
print("\n📊 Evaluating on Testing folder...")
test_gen = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'Testing'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
loss, acc = model.evaluate(test_gen, verbose=1)
print(f"🎯 Test accuracy: {acc:.4f} ({acc*100:.1f}%)")
