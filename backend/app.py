from flask import Flask, request, jsonify
import requests
import tempfile
import os
from deepfake_geometry_analysis import analyze_image_with_visuals, check_exif_metadata, analyze_face
import matplotlib
matplotlib.use('Agg')

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pytesseract
import re
from openai import OpenAI
import torch
import openai

app = Flask(__name__)

# Load CLIP model once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def openai_verify_content(text):
    response = openai.Moderation.create(input=text)  # <-- FIXED LINE
    result = response["results"][0]
    return {
        "harmful": result["flagged"],
        "categories": result["categories"],
        "clickbait": "violence" in result["categories"] or "sexual" in result["categories"],
        "accuracy": "inaccurate" if result["flagged"] else "accurate"
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_url = data['image_url']

    # Download image
    response = requests.get(image_url)
    if response.status_code != 200:
        return jsonify({"verdict": "ERROR", "reason": "Image fetch failed"})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        # Geometry + Visuals
        verdict_geom, reasons_geom, visuals = analyze_image_with_visuals(tmp_path)

        # CLIP classification
        clip_label, clip_conf = classify_with_clip(tmp_path)

        # OCR and OpenAI text classification
        extracted_text = pytesseract.image_to_string(Image.open(tmp_path))
        openai_result = {}
        if len(extracted_text) > 8 and re.search(r'[a-zA-Z]', extracted_text):
            openai_result = openai_verify_content(extracted_text)

        reasons = reasons_geom.copy()
        if openai_result:
            if openai_result.get('clickbait'):
                reasons.append("Clickbait detected")
            if openai_result.get('harmful'):
                reasons.append("Potentially harmful or misleading content")
            if openai_result.get('accuracy') == "inaccurate":
                reasons.append("Factual inaccuracy")

        if clip_label in ["violent", "clickbait"]:
            reasons.append(f"CLIP flagged as {clip_label} (conf={clip_conf:.2f})")

        flags = []
        if "violent" in clip_label:
            flags.append("blur_violent")
        if "clickbait" in clip_label:
            flags.append("blur_clickbait")

        final_verdict = "AI_GENERATED" if reasons else "REAL"

        return jsonify({
            "verdict": final_verdict,
            "reasons": reasons,
            "visuals": visuals,
            "text_analysis": openai_result,
            "clip_label": clip_label,
            "extracted_text": extracted_text,
            "flags": flags
        })

    finally:
        os.unlink(tmp_path)

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

def classify_with_clip(image_path):
    labels = ["normal", "violent", "clickbait", "fake"]
    image = Image.open(image_path)
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    top_idx = probs.argmax().item()
    return labels[top_idx], probs[0, top_idx].item()

if __name__ == "__main__":
    app.run(debug=True)
