import openai
import json
import re
from openai import OpenAI
# Configure OpenAI
client = OpenAI(api_key="my-key")  # Replace with your real key

import openai
import json
import re
from openai import OpenAI

# Setup your OpenAI key and client
#client = OpenAI(api_key="your-openai-api-key")  # Replace with your actual key

def openai_verify_content(extracted_text):
    system_prompt = "You are an expert fact-checker."
    user_prompt = f"""
Analyze the following text extracted from an image:

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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or gpt-4
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()

        print("🔍 Raw OpenAI response:")
        print(repr(content))

        # Clean code block if wrapped in markdown
        content = re.sub(r"^```(json)?", "", content)
        content = re.sub(r"```$", "", content)
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print("❌ JSON parsing failed.")
            print(f"Raw response: {content}")
            print(f"Error: {e}")
            return None

    except Exception as e:
        print(f"OpenAI Error: {e}")
        return None

if __name__ == '__main__':
    extracted_text = "The moon landing was faked."
    result = openai_verify_content(extracted_text)
    print("\n✅ Final Output:")
    print(result)
