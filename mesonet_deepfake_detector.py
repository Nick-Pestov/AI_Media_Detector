import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ExifTags
import pywt
import io
import os
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense

# -------------------------------------------
# (A) Face Forgery (Meso4)
# -------------------------------------------
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
        print(f"[!] Meso4 weights not found at {weights_path}.")
        return None
    model = build_meso4()
    model.load_weights(weights_path)
    print(f"[*] Loaded Meso4 pretrained from {weights_path}")
    return model


# -------------------------------------------
# (B) EXIF
# -------------------------------------------
def check_exif_metadata(image_path):
    """
    Return exif_info, suspicious_reason
    """
    exif_info = {}
    suspicious_reason = None
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is None:
                suspicious_reason = "No EXIF data."
                return exif_info, True
            else:
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_info[tag_name] = value
                # Example: if 'Software' in exif_info and 'Adobe' in exif_info['Software']:
                #     suspicious_reason = "Edited in Adobe software."
    except:
        suspicious_reason = "EXIF error reading."
    return exif_info, False


# -------------------------------------------
# (C) Face Detection + Meso4
# -------------------------------------------
def detect_and_crop_face(img_path, cascade_path="haarcascade_frontalface_default.xml", target_size=(256, 256)):
    if not os.path.exists(cascade_path):
        print(f"[!] Haar cascade not found: {cascade_path}")
        return None, None

    face_cascade = cv2.CascadeClassifier(cascade_path)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None, None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face_bgr = img_bgr[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, target_size)
    return face_rgb, (x, y, w, h)

def run_meso4_on_face(face_rgb, meso_weights="Meso4_DF.h5"):
    model = load_pretrained_mesonet(meso_weights)
    if model is None:
        return "NoModel", 0.0
    arr = face_rgb.astype(np.float32)/255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model(arr, training=False).numpy()[0, 0]
    if pred > 0.9:
        return "Fake", pred
    else:
        return "Real", 1.0 - pred

# -------------------------------------------
# (D) Wavelet / ELA / Pixel Anomaly / FFT
# -------------------------------------------


####################################################################################################
# Will not use these functions in the final analysis since unreliable. Yield lots of False Negatives

############# NOT USED #############

def wavelet_analysis(image_path):
    """
    Return the wavelet detail map (8-bit) and a detail_energy float
    """
    with Image.open(image_path).convert('L') as original:
        arr = np.array(original, dtype=np.float32)
    coeffs2 = pywt.dwt2(arr, 'db1')
    cA, (cH, cV, cD) = coeffs2

    detail_mag = np.sqrt(cH**2 + cV**2 + cD**2)
    detail_energy = float(detail_mag.mean())  # average detail magnitude

    # normalize for visualization
    detail_mag -= detail_mag.min()
    max_val = detail_mag.max()
    if max_val > 0:
        detail_mag /= max_val
    detail_map = (detail_mag * 255).astype(np.uint8)

    return detail_map, detail_energy

def ela_analysis(image_path, quality=90):
    """
    Return ELA image (PIL) and average_brightness
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
        ela_img = diff.point(lambda i: i * scale)

    # average brightness
    ela_np = np.array(ela_img.convert('L'), dtype=np.float32)
    avg_brightness = float(ela_np.mean())
    return ela_img, avg_brightness

def pixel_anomaly_map(image_pil, block_size=8):
    """
    Returns heatmap(0..1) and fraction_flagged(0..1).
    """
    img_np = np.array(image_pil.convert('RGB'), dtype=np.float32)
    H, W, _ = img_np.shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    flagged_pixels = 0
    total_pixels = H*W

    for yy in range(0, H, block_size):
        for xx in range(0, W, block_size):
            patch = img_np[yy:yy+block_size, xx:xx+block_size]
            if patch.size == 0:
                continue
            std_dev = np.std(patch)
            if std_dev < 4:
                # uniform
                heatmap[yy:yy+block_size, xx:xx+block_size] = 1.0
                flagged_pixels += (block_size*block_size)
            elif std_dev > 50:
                # super sharp
                heatmap[yy:yy+block_size, xx:xx+block_size] = 0.5
                flagged_pixels += (block_size*block_size)

    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val

    fraction_flagged = flagged_pixels / float(total_pixels)
    return heatmap, fraction_flagged

def fft_naturalness(image_path):
    """
    Returns a measure of how "natural" the frequency distribution is.
    We'll do:
      1) Convert to grayscale
      2) 2D FFT
      3) Compute radial average of magnitude
      4) Fit or approximate the slope => if slope is too "flat" or "steep", suspicious
    We'll just return the slope as a single float. (A real approach might be more elaborate.)
    """
    with Image.open(image_path).convert('L') as im:
        img_np = np.array(im, dtype=np.float32)

    # 2D FFT
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    # radial average
    H, W = mag.shape
    cy, cx = H//2, W//2
    rmax = int(np.hypot(cx, cy))
    radial_vals = []
    for r in range(1, rmax, 5):  # step radius by 5 for speed
        mask = (np.square(np.arange(H)[:,None]-cy) + np.square(np.arange(W)-cx)) 
        ring = (mask>=((r-2)**2)) & (mask<=(r+2)**2)
        ring_mag = mag[ring]
        if ring_mag.size > 0:
            radial_vals.append((r, ring_mag.mean()))
    # Fit log-log slope
    # y ~ 1/f => log(y) = -1*log(r) + const
    # We'll do a simple linear fit in log space
    if len(radial_vals)<2:
        return 0.0  # can't fit

    rs = np.array([x[0] for x in radial_vals], dtype=np.float32)
    vals = np.array([x[1] for x in radial_vals], dtype=np.float32)
    rs_log = np.log(rs+1e-8)
    vals_log = np.log(vals+1e-8)

    # Fit slope
    A = np.vstack([rs_log, np.ones_like(rs_log)]).T
    slope, intercept = np.linalg.lstsq(A, vals_log, rcond=None)[0]

    return slope  # Typically negative for natural images

# -------------------------------------------
# (E) Full Workflow With Heuristics
# -------------------------------------------
def analyze_image(
    image_path,
    face_cascade="haarcascade_frontalface_default.xml",
    meso_weights="Meso4_DF.h5"
):
    # We'll store reasons
    suspicious_reasons = []

    # 1) EXIF
    exif_info, exif_susp = check_exif_metadata(image_path)
    if exif_susp:
        suspicious_reasons.append(f"EXIF: {exif_susp}")

    # 2) Face detection => Meso4
    face_rgb, bbox = detect_and_crop_face(image_path, face_cascade)
    face_label, face_conf = "NoFace", 0.0
    if face_rgb is not None:
        face_label, face_conf = run_meso4_on_face(face_rgb, meso_weights)
        if face_label == "Fake" and face_conf > 0.6:
            suspicious_reasons.append(f"Meso4: Fake face conf={face_conf:.2f}")

    # Not used since unreliable. Looks cool though, and visualizes heatmaps!
    """
    # 3) Wavelet => detail energy
    wave_map, detail_energy = wavelet_analysis(image_path)
    # Heuristic: normal images might have detail_energy ~ 5..30
    # TOTALLY depends on resolution. We'll do a naive check
    if detail_energy < 4 or detail_energy > 50:
        suspicious_reasons.append(f"Wavelet detail energy out of range ({detail_energy:.2f})")

    # 4) ELA => average brightness
    ela_img, ela_brightness = ela_analysis(image_path, quality=90)
    # Heuristic: normal ELA brightness might be ~10..30 range 
    # Very low => suspiciously uniform, Very high => suspicious recompression
    if ela_brightness < 5 or ela_brightness > 40:
        suspicious_reasons.append(f"ELA brightness suspicious ({ela_brightness:.2f})")

    # 5) Pixel anomaly => fraction flagged
    anomaly_map, frac_flagged = pixel_anomaly_map(ela_img, block_size=8)
    # If > 0.2 => suspicious
    if frac_flagged > 0.2:
        suspicious_reasons.append(f"Pixel anomaly fraction high ({frac_flagged:.2f})")
    """

    # 6) FFT => slope
    slope = fft_naturalness(image_path)
    # Typical slopes for natural images might be around ~-1 ~-2
    # If slope > -0.5 or slope < -2.5 => suspicious
    if slope > -0.5 or slope < -2.5:
        suspicious_reasons.append(f"FFT slope out of normal range (slope={slope:.2f})")

    # Determine final label
    final_label = "PASS" if len(suspicious_reasons)==0 else "FLAGGED"
    print(f"Final Verdict: {final_label}")
    if final_label=="FLAGGED":
        for r in suspicious_reasons:
            print(" -", r)

    # Visualization
    # Show Original, wavelet map, ELA, anomaly
    original_bgr = cv2.imread(image_path)
    show_plots(original_bgr, bbox, face_label, face_conf)

def show_plots(orig_bgr, face_bbox, face_label, face_conf):
    # Draw face box if any
    if orig_bgr is not None and face_bbox is not None and face_label != "NoFace":
        x,y,w,h = face_bbox
        cv2.rectangle(orig_bgr, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(orig_bgr, f"{face_label}({face_conf:.2f})", 
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

    images = []
    titles = []

    if orig_bgr is not None:
        images.append(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        titles.append("Original")

    """
    if wave_map is not None:
        images.append(wave_map)
        titles.append("Wavelet Detail")

    if ela_img is not None:
        ela_np = np.array(ela_img.convert('RGB'))
        images.append(ela_np)
        titles.append("ELA Map")

    if anomaly_map is not None:
        an_np = (anomaly_map*255).astype(np.uint8)
        an_vis = cv2.applyColorMap(an_np, cv2.COLORMAP_JET)
        an_vis = cv2.cvtColor(an_vis, cv2.COLOR_BGR2RGB)
        images.append(an_vis)
        titles.append("Pixel Anomaly")

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5*n,5))
    if n==1:
        axes=[axes]

    for i,ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    """

# -------------------------------------------
# (F) Run
# -------------------------------------------
if __name__=="__main__":
    #IMAGE_PATH = "AI_generated2.png"  # or "some_image.png"
    IMAGE_PATH = "real.jpg"
    CASCADE_PATH = "haarcascade_frontalface_default.xml"
    MESO_WEIGHTS= "Meso4_DF.h5"

    analyze_image(IMAGE_PATH, CASCADE_PATH, MESO_WEIGHTS)
