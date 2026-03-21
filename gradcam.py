import numpy as np
import tensorflow as tf
import cv2
import base64
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

_grad_model = None

def _build_grad_model(model):
    """
    Build one connected graph:
      input_2 → efficientnetb0 → top_activation → Dense head → softmax
    Outputs: [top_conv activations, final predictions]
    """
    # Get EfficientNet sub-model
    efficientnet = model.get_layer("efficientnetb0")

    # top_activation is the last spatial feature map (after top_conv + top_bn)
    # Using top_activation instead of top_conv gives slightly better heatmaps
    top_act_output = efficientnet.get_layer("top_activation").output

    # Rebuild the classifier head on top of efficientnet.output
    # by calling each remaining outer layer in order
    x = efficientnet.output
    for layer in model.layers:
        if layer.name in ("input_2", "efficientnetb0"):
            continue
        x = layer(x)

    # Both outputs share efficientnet.input as the single entry point
    grad_model = tf.keras.models.Model(
        inputs=efficientnet.input,
        outputs=[top_act_output, x]
    )
    return grad_model


def generate_gradcam(model, pil_image, class_index):
    global _grad_model
    if _grad_model is None:
        _grad_model = _build_grad_model(model)

    # Prepare image
    img_resized = pil_image.resize((224, 224)).convert("RGB")
    img_array = np.array(img_resized, dtype=np.float32)
    img_preprocessed = preprocess_input(img_array.copy())
    img_tensor = tf.cast(np.expand_dims(img_preprocessed, 0), tf.float32)

    # Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = _grad_model(img_tensor)
        tape.watch(conv_outputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Grad-CAM: gradients are None.")

    # Heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_out = conv_outputs[0].numpy()
    for i in range(pooled_grads.shape[0]):
        conv_out[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # Overlay
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_array.astype(np.uint8), 0.55, colored, 0.45, 0)

    buf = BytesIO()
    Image.fromarray(superimposed).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"