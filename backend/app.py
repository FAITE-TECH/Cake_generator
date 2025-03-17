from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import os
import io
from dotenv import load_dotenv
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# ✅ Apply enhanced CORS configuration
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    print("Error: API Key not found!")

HF_MODEL = "CompVis/stable-diffusion-v1-4"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {api_key}"}

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = (f"A {data['shape']} cake made with {data['flour_type']} flour, "
                  f"designed for a {data['function_type']}. It has {data['icing_type']} icing "
                  f"with {data['filling']} filling. The cake follows a {data['color_theme']} color theme, "
                  f"topped with {data['toppings']}. It has {data['layers']} layers and a {data['texture']} texture. "
                  f"Decorations include {data['decorations']}. The cake should be visually appealing.")

        response = requests.post(HF_URL, headers=HEADERS, json={"inputs": prompt})

        if response.status_code == 200:
            image_bytes = response.content
            image = Image.open(io.BytesIO(image_bytes))
            image_path = "generated_image.png"
            image.save(image_path)

            return send_file(image_path, mimetype="image/png")

        else:
            return jsonify({"error": "Failed to generate image", "details": response.text}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # ✅ Ensures access from different interfaces
