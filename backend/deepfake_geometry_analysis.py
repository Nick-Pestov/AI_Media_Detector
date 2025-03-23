import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from mesonet_deepfake_detector import detect_and_crop_face, run_meso4_on_face, check_exif_metadata
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg') 

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

def analyze_image_with_visuals(image_path):
    reasons = []
    visuals = {}
    flags = []

    # EXIF Check
    _, exif_missing = check_exif_metadata(image_path)
    if exif_missing:
        reasons.append("Missing EXIF metadata (may indicate manipulation).")
        flags.append("exif_missing")

    # Frequency Analysis
    radial_profile, freq_slope = analyze_frequency_distribution(image_path)
    if freq_slope is not None:
        fig = show_frequency_profile(radial_profile)
        visuals["fft_analysis"] = fig_to_base64(fig)

        # Only flag if very unnatural
        if not (-2.5 < freq_slope < -0.5):
            reasons.append(f"Unnatural frequency distribution (slope={freq_slope:.2f})")
            flags.append("fft_suspicious")
        else:
            flags.append("fft_visual_only")  # Will show visual but not trigger alert

    # Blur Analysis
    laplacian_vars, avg_blur = local_blur_analysis(image_path)
    if avg_blur < 10:
        reasons.append("Image is unnaturally blurry or AI-generated.")
        fig = plt.figure()
        plt.hist(laplacian_vars, bins=20, color='blue')
        plt.title("Local Blur Variance")
        plt.xlabel("Variance")
        plt.ylabel("Frequency")
        visuals["blur_analysis"] = fig_to_base64(fig)

    # Horizon Geometry
    horizon_lines, _ = horizon_checker(image_path)
    suspicious_geometry = analyze_horizon_heuristics(horizon_lines)
    if suspicious_geometry:
        reasons.append("Suspicious horizon geometry.")
        img_color = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR)
        for (x1, y1, x2, y2, _) in horizon_lines:
            cv2.line(img_color, (x1, y1), (x2, y2), (0,0,255), 2)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        ax.set_title("Detected Horizon Lines")
        ax.axis('off')
        visuals["horizon_analysis"] = fig_to_base64(fig)
        plt.title("Detected Horizon Lines")
        plt.axis('off')
        visuals["horizon_analysis"] = fig_to_base64(fig)

    # Face Forgery Detection
    current_dir = os.path.dirname(os.path.abspath(__file__))
    face_rgb, bbox = detect_and_crop_face(image_path, os.path.join(current_dir, "haarcascade_frontalface_default.xml"))
    if face_rgb is not None:
        face_label, face_conf = run_meso4_on_face(face_rgb, meso_weights=os.path.join(current_dir, "Meso4_DF.h5"))
        if face_label == "Fake" and face_conf > 0.6:
            reasons.append(f"Face forgery detected with confidence {face_conf:.2f}.")
            flags.append("face_forgery")
            fig = plt.figure()
            plt.imshow(face_rgb)
            plt.title(f"Detected Face: {face_label} ({face_conf:.2f})")
            plt.axis('off')
            visuals["face_analysis"] = fig_to_base64(fig)

    verdict = "AI_GENERATED" if reasons else "REAL"
    return verdict, reasons, visuals, flags

# FFT analysis

def analyze_frequency_distribution(image_path):
    try:
        with Image.open(image_path).convert('L') as im:
            img_np = np.array(im, dtype=np.uint8)
    except Exception as e:
        print("❌ Unsupported image format for file, cannot analyze it T_T")
        return None, None
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    H, W = magnitude_spectrum.shape
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

    radial_sum = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    radial_count = np.bincount(r.ravel())
    radial_profile = radial_sum / (radial_count + 1e-8)

    r_vals = np.arange(1, len(radial_profile))
    profile = radial_profile[1:]
    log_r = np.log(r_vals + 1e-8)
    log_profile = np.log(profile + 1e-8)
    A = np.vstack([log_r, np.ones_like(log_r)]).T
    slope, _ = np.linalg.lstsq(A, log_profile, rcond=None)[0]

    return radial_profile, slope

def show_frequency_profile(radial_profile):
    r_vals = np.arange(1, len(radial_profile))
    fig, ax = plt.subplots()
    ax.loglog(r_vals, radial_profile[1:], label='Radial profile')
    ax.set_xlabel('Frequency (radius)')
    ax.set_ylabel('Average magnitude')
    ax.set_title('Frequency Distribution (Radial Average)')
    ax.legend()
    return fig

# Blur analysis

def local_blur_analysis(image_path, block_size=64):
    with Image.open(image_path).convert('L') as im:
        img_np = np.array(im, dtype=np.uint8)
    H, W = img_np.shape
    laplacian_vars = []

    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            patch = img_np[y:y+block_size, x:x+block_size]
            if patch.shape[0] < block_size or patch.shape[1] < block_size:
                continue
            lap = cv2.Laplacian(patch, cv2.CV_64F)
            var = lap.var()
            laplacian_vars.append(var)

    avg_var = np.mean(laplacian_vars) if laplacian_vars else 0.0
    return laplacian_vars, avg_var

# Horizon detection

def horizon_checker(image_path, edge_thresh1=50, edge_thresh2=150, hough_thresh=100):
    with Image.open(image_path).convert('L') as im:
        img_np = np.array(im, dtype=np.uint8)
    edges = cv2.Canny(img_np, edge_thresh1, edge_thresh2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thresh, minLineLength=100, maxLineGap=10)
    horizon_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 10 or abs(angle) > 170:
                horizon_lines.append((x1, y1, x2, y2, angle))
    return horizon_lines, edges

def analyze_horizon_heuristics(horizon_lines):
    angles = [angle for (_, _, _, _, angle) in horizon_lines]
    y_coords = [(y1 + y2) // 2 for (_, y1, _, y2, _) in horizon_lines]

    angle_std = np.std(angles) if angles else 0
    y_range = np.ptp(y_coords) if y_coords else 0
    num_lines = len(horizon_lines)

    print(f"[+] Horizon Heuristics:")
    print(f"  - Num horizontal lines: {num_lines}")
    print(f"  - Angle std deviation : {angle_std:.2f}")
    print(f"  - Vertical spread      : {y_range} pixels")

    suspicious = False
    if num_lines < 5:
        print("  ⚠️ Few horizontal lines detected")
        suspicious = True
    if angle_std > 5:
        print("  ⚠️ Inconsistent horizon angles (std > 5)")
        suspicious = True
    if y_range < 100:
        print("  ⚠️ Horizon lines bunched (spread < 100 px)")
        suspicious = True

    return suspicious
# ------------------------------
# (5) MesoNet Analysis (Face forgery) <-- not the most reliable so it has a high threshold, but still captures very high values well
# ------------------------------

def analyze_face(image_path, face_cascade="haarcascade_frontalface_default.xml", meso_weights="Meso4_DF.h5"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    face_cascade_path = os.path.join(current_dir, face_cascade)
    meso_weights_path = os.path.join(current_dir, meso_weights)

    face_rgb, bbox = detect_and_crop_face(image_path, face_cascade_path)
    face_label, face_conf = "NoFace", 0.0
    suspicious = False
    if face_rgb is not None:
        face_label, face_conf = run_meso4_on_face(face_rgb, meso_weights=meso_weights_path)
        if face_label == "Fake" and face_conf > 0.6:
            print(f"Meso4: Fake face conf={face_conf:.2f}")
            suspicious = True
    return suspicious


# ------------------------------
# Full Analysis & Visualization
# ------------------------------
def analyze_image(image_path, save_plots=False):
    radial_profile, freq_slope = analyze_frequency_distribution(image_path)
    _, avg_blur = local_blur_analysis(image_path)
    horizon_lines, _ = horizon_checker(image_path)
    suspicious_geometry = analyze_horizon_heuristics(horizon_lines)
    face_suspicious = analyze_face(image_path)
    _, exif_missing = check_exif_metadata(image_path)

    reasons = []
    if exif_missing:
        reasons.append("Missing EXIF data.")
    if suspicious_geometry:
        reasons.append("Suspicious geometry detected.")
    if face_suspicious:
        reasons.append("Face forgery detected.")
    if avg_blur < 10:
        reasons.append("Unnaturally blurry or AI regions.")

    if reasons:
        return "AI_GENERATED", "; ".join(reasons)
    return "REAL", "Image seems authentic."

def analyze_image2(image_path, save_plots=False):
    print("=== Frequency Analysis ===")
    radial_profile, freq_slope = analyze_frequency_distribution(image_path)
    print(f"Frequency slope: {freq_slope:.2f} (Natural: -1 to -2)")
    show_frequency_profile(radial_profile, save_path="freq_profile.png" if save_plots else None)

    print("\n=== Blur Analysis ===")
    _, avg_blur = local_blur_analysis(image_path)
    print(f"Avg Laplacian variance: {avg_blur:.2f} (Low = blurry or AI regions)")

    print("\n=== Horizon Analysis ===")
    horizon_lines, _ = horizon_checker(image_path)
    if horizon_lines:
        print(f"Detected {len(horizon_lines)} horizon lines")
    else:
        print("No clear horizontal lines found.")

    suspicious = analyze_horizon_heuristics(horizon_lines)

    img_color = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR)
    if horizon_lines:
        for (x1, y1, x2, y2, _) in horizon_lines:
            cv2.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("Edges with Horizon Lines")
    plt.axis('off')
    if save_plots:
        plt.savefig("horizon_lines.png")
    else:
        plt.show()

    print("\n------------------------")
    _, missing_tag = check_exif_metadata(image_path)
    if missing_tag:
        print("🟡 Does not contain an EXIF tag, authenticity cannot be verified.")
    if suspicious:
        print("🚩 Verdict: POSSIBLY AI-GENERATED (suspicious geometry)")
    face_passed = analyze_face(image_path)
    if face_passed:
        print("🚩 Verdict: POSSIBLY AI-GENERATED (suspicious face)")
    if not suspicious and not face_passed:
        print("✅ Verdict: Likely REAL")
    print("------------------------\n")

if __name__ == "__main__":
    IMAGE_PATH = "./backend/image8.avif"
    analyze_image2(IMAGE_PATH)
