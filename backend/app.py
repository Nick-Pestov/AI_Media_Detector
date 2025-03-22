from flask import Flask, request, jsonify
import requests
import tempfile
import os
import matplotlib
matplotlib.use('Agg')

from deepfake_geometry_analysis import analyze_image_with_visuals, check_exif_metadata, analyze_face
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pytesseract
import re
from openai import OpenAI
import torch
from openAI import openai_verify_content

# Init Flask
app = Flask(__name__)

# Load CLIP once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Setup OpenAI
client = OpenAI(api_key="your-api-key")

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
        # Step 1: Deepfake Geometry & Visual
        verdict_geom, reasons_geom, visuals = analyze_image_with_visuals(tmp_path)
        
        # Step 2: CLIP Image Labeling
        clip_label, clip_conf = classify_with_clip(tmp_path)

        # Step 3: Extract Text from Image
        extracted_text = pytesseract.image_to_string(Image.open(tmp_path))
        if len(extracted_text) > 8 and re.search(r'[a-zA-Z]', extracted_text):
            openai_result = openai_verify_content(extracted_text)

        reasons = reasons_geom.copy()
        if openai_result:
            if openai_result['clickbait']:
                reasons.append("Clickbait detected")
            if openai_result['harmful']:
                reasons.append("Potentially harmful or misleading content")
            if openai_result['accuracy'] == "inaccurate":
                reasons.append("Factual inaccuracy")
        
        if clip_label in ["violent", "clickbait"]:
            reasons.append(f"CLIP flagged as {clip_label} (conf={clip_conf:.2f})")

        final_verdict = "AI_GENERATED" if reasons else "REAL"

        return jsonify({
            "verdict": final_verdict,
            "reasons": reasons,
            "visuals": visuals,
            "text_analysis": openai_result,
            "clip_label": clip_label,
            "extracted_text": extracted_text
        })

    finally:
        os.unlink(tmp_path)

# -------------------------
# CLIP Classification
# -------------------------
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
