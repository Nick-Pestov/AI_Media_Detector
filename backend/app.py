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
    response = client.moderations.create(input=text) # <-- FIXED LINE
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
    response = requests.get(image_url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36'
    })
    if response.status_code != 200:
        return jsonify({"verdict": "ERROR", "reason": "Image fetch failed"})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        # Geometry + Visuals
        verdict_geom, reasons_geom, visuals, flags_geom = analyze_image_with_visuals(tmp_path)

        # CLIP classification
        clip_label, clip_conf = classify_with_clip(tmp_path)

        # OCR and OpenAI text classification
        extracted_text = pytesseract.image_to_string(Image.open(tmp_path))
        openai_result = {}
        if len(extracted_text) > 8 and re.search(r'[a-zA-Z]', extracted_text):
            openai_result = openai_verify_content(extracted_text)

        reasons = reasons_geom.copy()
        flags = flags_geom.copy()

        if openai_result:
            if openai_result.get('clickbait'):
                reasons.append("Clickbait detected")
                flags.append("clickbait")
            if openai_result.get('harmful'):
                reasons.append("Potentially harmful or misleading content")
            if openai_result.get('accuracy') == "inaccurate":
                reasons.append("Factual inaccuracy")

        if clip_label in ["violent", "clickbait"]:
            reasons.append(f"CLIP flagged as {clip_label} (conf={clip_conf:.2f})")
            flags.append(f"blur_{clip_label}")

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
