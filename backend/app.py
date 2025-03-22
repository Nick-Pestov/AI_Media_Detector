from flask import Flask, request, jsonify
from deepfake_geometry_analysis import analyze_image
import requests
import tempfile
import os

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

    verdict, reason = analyze_image_api(tmp_path)
    os.unlink(tmp_path)

    return jsonify({"verdict": verdict, "reason": reason})

def analyze_image_api(image_path):
    from deepfake_geometry_analysis import (
        analyze_frequency_distribution,
        local_blur_analysis,
        horizon_checker,
        analyze_horizon_heuristics,
        analyze_face,
        check_exif_metadata
    )

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

if __name__ == "__main__":
    app.run(debug=True)
