# predict.py — Run Inference on a New MRI Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import load_img, img_to_array

CLASSES = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor',
    2: 'No Tumor',
    3: 'Pituitary Tumor'
}

def predict_tumor(img_path, model_path='best_model.h5'):
    """Load model and predict tumor class from MRI image."""

    # Load trained model
    model = load_model(model_path)

    # Preprocess image (same as training)
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get predictions
    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100

    # Display results
    print(f"\n🧠 Prediction: {CLASSES[predicted_class]}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"\nAll Class Probabilities:")
    for i, cls in CLASSES.items():
        bar = '█' * int(predictions[i] * 30)
        print(f"  {cls:20s} {bar} {predictions[i]*100:.1f}%")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img); plt.title('MRI Scan'); plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.bar(list(CLASSES.values()), predictions, color=['#ff4f6d','#ffaa00','#00ff99','#00d4ff'])
    plt.title('Class Probabilities'); plt.ylim(0,1); plt.xticks(rotation=15)
    plt.tight_layout(); plt.show()

    return CLASSES[predicted_class], confidence

# ── Run it ────────────────────────────────────────────────
if __name__ == '__main__':
    result, conf = predict_tumor('test_mri.jpg')
