from flask import Flask, request, jsonify
import requests
import tempfile
import os
from deepfake_geometry_analysis import analyze_image_with_visuals
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_url = data['image_url']
    response = requests.get(image_url)
    if response.status_code != 200:
        return jsonify({"verdict": "ERROR", "reason": "Image fetch failed"})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    verdict, reasons, visuals = analyze_image_with_visuals(tmp_path)
    os.unlink(tmp_path)

    return jsonify({"verdict": verdict, "reasons": reasons, "visuals": visuals})

def analyze_image_api(image_path):
    from deepfake_geometry_analysis import (
        analyze_frequency_distribution,
        local_blur_analysis,
        horizon_checker,
        analyze_horizon_heuristics,
        analyze_face,
        check_exif_metadata
    )

    reasons = []

    # EXIF Check
    _, exif_missing = check_exif_metadata(image_path)
    if exif_missing:
        reasons.append("Missing EXIF data")

    # Geometry checks
    _, freq_slope = analyze_frequency_distribution(image_path)
    if not -2.5 < freq_slope < -0.5:
        reasons.append(f"Unnatural frequency distribution (slope={freq_slope:.2f})")

    _, avg_blur = local_blur_analysis(image_path)
    if avg_blur < 10:
        reasons.append("Unnaturally blurry or AI regions")

    horizon_lines, _ = horizon_checker(image_path)
    suspicious_geometry = analyze_horizon_heuristics(horizon_lines)
    if suspicious_geometry:
        reasons.append("Suspicious geometry detected")

    face_suspicious = analyze_face(image_path)
    if face_suspicious:
        reasons.append("Face forgery detected")

    if reasons:
        return "AI_GENERATED", "; ".join(reasons)
    else:
        return "REAL", "Image seems authentic."

if __name__ == "__main__":
    app.run(debug=True)
