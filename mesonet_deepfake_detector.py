import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ExifTags
import pywt       # We'll use "pywt" in code
import io
import os
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense

# -------------------------------------------------
# 0) Utility: MesoNet Model (Face Forgery Detection)
# -------------------------------------------------
def build_meso4():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(8, (3,3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    x = Conv2D(8, (5,5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    x = Conv2D(16, (5,5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same')(x)

    x = Conv2D(16, (5,5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4,4), padding='same')(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs, output)

def load_pretrained_mesonet(weights_path="Meso4_DF.h5"):
    if not os.path.exists(weights_path):
        print(f"[!] Meso4 weights not found at {weights_path}. Please place Meso4_DF.h5 alongside this script.")
        return None
    model = build_meso4()
    model.load_weights(weights_path)
    print(f"[*] Loaded Meso4 pretrained weights from {weights_path}")
    return model

# -------------------------------------------------
# 1) EXIF & Metadata Checking (Lightweight)
# -------------------------------------------------
def check_exif_metadata(image_path):
    """
    Prints some EXIF info if present (e.g., camera make/model).
    Large inconsistencies can be a manipulation flag.
    """
    if not os.path.exists(image_path):
        print("[!] Image does not exist.")
        return

    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()  # Basic PIL EXIF
            if exif_data is None:
                print("[*] No EXIF data found.")
            else:
                print("[*] EXIF data:")
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                    print(f"    {tag_name}: {value}")
    except Exception as e:
        print(f"[!] EXIF read error: {e}")

# -------------------------------------------------
# 2) Face Detection & Cropping (if face found)
# -------------------------------------------------
def detect_and_crop_face(image_path, cascade_path="haarcascade_frontalface_default.xml", target_size=(256, 256)):
    if not os.path.exists(cascade_path):
        print(f"[!] Haar cascade file not found: {cascade_path}")
        return None, None

    face_cascade = cv2.CascadeClassifier(cascade_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("[!] Could not read image for face detection.")
        return None, None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face_bgr = img_bgr[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, target_size)
    return face_rgb, (x, y, w, h)

def run_meso4_on_face(face_img, meso_weights="Meso4_DF.h5"):
    """
    Loads Meso4 if not already loaded, runs on a single face crop.
    Return label "Fake"/"Real" and confidence.
    """
    model = load_pretrained_mesonet(meso_weights)
    if model is None:
        return "NoModel", 0.0

    arr = face_img.astype(np.float32)/255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model(arr, training=False).numpy()[0, 0]
    if pred > 0.5:
        return "Fake", pred
    else:
        return "Real", (1.0 - pred)

# -------------------------------------------------
# 3) Wavelet-based Analysis
# -------------------------------------------------
def wavelet_analysis(image_path):
    """
    Simple wavelet decomposition using PyWavelets (db1).
    Returns an 8-bit detail magnitude image.
    """
    with Image.open(image_path).convert('L') as original:
        arr = np.array(original, dtype=np.float32)
    coeffs2 = pywt.dwt2(arr, 'db1')
    cA, (cH, cV, cD) = coeffs2
    detail_mag = np.sqrt(cH**2 + cV**2 + cD**2)
    detail_mag -= detail_mag.min()
    max_val = detail_mag.max()
    if max_val != 0:
        detail_mag /= max_val
    detail_mag_img = (detail_mag * 255).astype(np.uint8)
    return detail_mag_img

# -------------------------------------------------
# 4) Error Level Analysis (ELA)
# -------------------------------------------------
def perform_ela(image_path, quality=90):
    """
    Resaves the image at 'quality' as JPEG, compares to original.
    Returns a PIL Image (the difference map).
    """
    with Image.open(image_path).convert('RGB') as original:
        buffer = io.BytesIO()
        original.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)

        diff = ImageChops.difference(original, resaved)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 0
        scale = 255.0 / max_diff if max_diff != 0 else 1.0
        diff = diff.point(lambda i: i * scale)
    return diff

# -------------------------------------------------
# 5) Naive Pixel Anomaly
# -------------------------------------------------
def pixel_anomaly_map(image_pil, block_size=8):
    """
    Blockwise std dev check:
      - Very low std => suspiciously uniform region
      - Very high std => suspiciously sharp region
    Returns a 0..1 heatmap as a np.float32 array
    """
    img_np = np.array(image_pil.convert('RGB'), dtype=np.float32)
    H, W, C = img_np.shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    for yy in range(0, H, block_size):
        for xx in range(0, W, block_size):
            patch = img_np[yy:yy+block_size, xx:xx+block_size, :]
            if patch.size == 0:
                continue
            std_dev = np.std(patch)
            # Simple thresholds
            if std_dev < 4:
                heatmap[yy:yy+block_size, xx:xx+block_size] = 1.0
            elif std_dev > 50:
                heatmap[yy:yy+block_size, xx:xx+block_size] = 0.5

    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val
    return heatmap

# -------------------------------------------------
# 6) Visualization
# -------------------------------------------------
def visualize_results(original_bgr, face_bbox, face_label_str, face_conf,
                      wavelet_map, ela_map, anomaly_heatmap):
    """
    Displays up to 4 subplots:
      - Original (with face box if found)
      - Wavelet detail
      - ELA
      - Anomaly heatmap
    """
    if original_bgr is None:
        print("[!] No original image to display.")
        return

    # Draw face box
    if face_bbox is not None:
        x, y, w, h = face_bbox
        cv2.rectangle(original_bgr, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(original_bgr, f"{face_label_str}({face_conf:.2f})", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # We'll show up to 4 images
    # 1) Original
    # 2) Wavelet
    # 3) ELA
    # 4) Pixel anomaly
    n_images = 1
    if wavelet_map is not None:
        n_images += 1
    if ela_map is not None:
        n_images += 1
    if anomaly_heatmap is not None:
        n_images += 1

    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    if n_images == 1:
        axes = [axes]

    idx = 0

    # Original
    axes[idx].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[idx].set_title("Original")
    axes[idx].axis('off')
    idx += 1

    # Wavelet
    if wavelet_map is not None:
        axes[idx].imshow(wavelet_map, cmap='gray')
        axes[idx].set_title("Wavelet Detail")
        axes[idx].axis('off')
        idx += 1

    # ELA
    if ela_map is not None:
        ela_np = np.array(ela_map.convert('RGB'))
        axes[idx].imshow(ela_np)
        axes[idx].set_title("ELA Map")
        axes[idx].axis('off')
        idx += 1

    # Anomaly
    if anomaly_heatmap is not None:
        anomaly_vis = (anomaly_heatmap * 255).astype(np.uint8)
        anomaly_vis = cv2.applyColorMap(anomaly_vis, cv2.COLORMAP_JET)
        axes[idx].imshow(cv2.cvtColor(anomaly_vis, cv2.COLOR_BGR2RGB))
        axes[idx].set_title("Pixel Anomaly")
        axes[idx].axis('off')
        idx += 1

    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# 7) Main Workflow
# -------------------------------------------------
def analyze_image(image_path, 
                  cascade_path="haarcascade_frontalface_default.xml",
                  meso_weights="Meso4_DF.h5",
                  do_wavelet=True,
                  do_ela=True,
                  do_anomaly=True,
                  check_exif=True):
    """
    Analyzes an image with multiple steps in order of quickness:
      1. EXIF check (quick)
      2. Face detection -> Meso4 if found
      3. Wavelet analysis
      4. ELA
      5. Naive pixel anomaly
    Then visualize.
    """
    if not os.path.exists(image_path):
        print(f"[!] No file at {image_path}")
        return

    print(f"Analyzing: {image_path}")

    # 1) EXIF
    if check_exif:
        print("[*] Checking EXIF/metadata (quick).")
        check_exif_metadata(image_path)
        print("-"*40)

    # 2) Face detection -> if found, run Meso4
    print("[*] Trying face detection (quick) ...")
    face_img, bbox = detect_and_crop_face(image_path, cascade_path=cascade_path)
    face_label_str, face_conf = "NoFace", 0.0
    if face_img is not None:
        print("[*] Face found. Running Meso4 ...")
        face_label_str, face_conf = run_meso4_on_face(face_img, meso_weights)
        print(f"    => {face_label_str} (conf={face_conf:.2f})")
    else:
        print("[!] No face found. Skipping Meso4.")
    print("-"*40)

    # 3) Wavelet
    wavelet_map = None
    if do_wavelet:
        print("[*] Wavelet analysis ...")
        wavelet_map = wavelet_analysis(image_path)
    else:
        print("[*] Skipping wavelet analysis.")

    # 4) ELA
    ela_img = None
    if do_ela:
        print("[*] Performing ELA (can be heavier).")
        ela_img = perform_ela(image_path, quality=90)
    else:
        print("[*] Skipping ELA.")

    # 5) Pixel anomaly on either ELA or original
    anomaly_heatmap = None
    if do_anomaly:
        print("[*] Checking pixel-level anomalies.")
        if ela_img is not None:
            anomaly_heatmap = pixel_anomaly_map(ela_img, block_size=8)
        else:
            # If ELA was skipped, do anomaly on original
            with Image.open(image_path) as pil_img:
                anomaly_heatmap = pixel_anomaly_map(pil_img, block_size=8)
    else:
        print("[*] Skipping anomaly map.")

    # Load original (for final display)
    original_bgr = cv2.imread(image_path)

    # Visualization
    visualize_results(
        original_bgr,
        bbox,
        face_label_str,
        face_conf,
        wavelet_map,
        ela_img,
        anomaly_heatmap
    )

# -------------------------------------------------
# 8) Manual Pixel-Level Inspection (If you want)
# -------------------------------------------------
def show_zoomed_region(image_path, x, y, region_size=50):
    """
    Crop around (x,y) => scale up => display.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("[!] Cannot open image for manual inspection.")
        return
    H, W, _ = img.shape
    x1 = max(0, x - region_size//2)
    y1 = max(0, y - region_size//2)
    x2 = min(W, x1 + region_size)
    y2 = min(H, y1 + region_size)
    cropped = img[y1:y2, x1:x2]

    # 10x nearest neighbor
    zoomed = cv2.resize(cropped, None, fx=10.0, fy=10.0, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Zoomed Region", zoomed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------------------------
# 9) Run from CLI
# -------------------------------------------------
if __name__ == "__main__":
    # Example usage:
    # Put Meso4_DF.h5 + haarcascade_frontalface_default.xml in same folder
    # Then run: python script.py
    IMAGE_PATH = "deep_fake.webp"  # or "photoshopped.png", "real.jpg", etc.
    CASCADE_PATH = "haarcascade_frontalface_default.xml"
    MESO_WEIGHTS = "Meso4_DF.h5"

    analyze_image(
        image_path=IMAGE_PATH,
        cascade_path=CASCADE_PATH,
        meso_weights=MESO_WEIGHTS,
        do_wavelet=True,
        do_ela=True,
        do_anomaly=True,
        check_exif=True
    )

    # For manual zoom in on suspicious coordinates:
    # show_zoomed_region(IMAGE_PATH, x=100, y=50, region_size=30)
