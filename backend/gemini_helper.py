import google.generativeai as genai
import json

# Configure Gemini
genai.configure(api_key='WWNWW7FXVN0NXY44')

def gemini_verify_content(extracted_text):
    prompt = f"""
    You are an expert fact-checker. Analyze the following text extracted from an image:
    
    "{extracted_text}"

    Clearly determine if it contains:
    1. Factually inaccurate claims
    2. Clickbait-style or misleading language
    3. Potentially harmful misinformation

    Respond in JSON:
    {{
        "accuracy": "accurate"/"inaccurate"/"unclear",
        "clickbait": true/false,
        "harmful": true/false,
        "reasoning": "Explain clearly your decision in 2-3 sentences.",
        "sources": ["source1", "source2", "source3"] (include only if inaccurate or harmful)
    }}
    """

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        json_response = response.text
        return json.loads(json_response)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None
