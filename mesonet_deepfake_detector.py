import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense

def build_meso4():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(8, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv2d_3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), padding='same')(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    return Model(inputs, output)

def load_pretrained_mesonet(weights_path="Meso4_DF.h5"):
    model = build_meso4()
    model.load_weights(weights_path)
    print(f"[*] Loaded Meso4 pretrained weights from {weights_path}")
    return model

def compute_gradcam_keras(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.input], 
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy() if hasattr(conv_outputs[0], 'numpy') else conv_outputs[0]
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i].numpy()
    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()
    return heatmap

def overlay_heatmap_on_image(heatmap, original_bgr, alpha=100.8):
    heatmap = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap_color, alpha, original_bgr, 1 - alpha, 0)

def predict_and_visualize(image_path, model, target_layer="conv2d_3"):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Couldn't read image at {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model(arr, training=False).numpy()[0, 0]
    label_str = "Fake" if pred > 0.5 else "Real"
    confidence = pred if pred > 0.5 else 1 - pred
    print(f"[*] Model Prediction: {label_str} (confidence = {confidence:.4f})")

    heatmap = compute_gradcam_keras(model, arr, layer_name=target_layer)
    overlay_img = overlay_heatmap_on_image(heatmap, img_bgr, alpha=0.7)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image - {label_str}")
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Suspicious Regions" if label_str == "Fake" else "Model Focus")
    plt.axis('off')
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    model = load_pretrained_mesonet("Meso4_DF.h5")
    test_image_path = "deep_fake.webp"
    predict_and_visualize(test_image_path, model, target_layer="conv2d_3")
