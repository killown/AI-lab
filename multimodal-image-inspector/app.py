"""Flask application providing a local interface and API endpoint
to analyze any uploaded or local image using qwen2.5vl:7b.
"""

import base64
from flask import Flask, render_template, request, jsonify
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

app = Flask(__name__)

VISION_MODEL = "qwen2.5vl:7b"
llm_vision = ChatOllama(model=VISION_MODEL, temperature=0.1)


def encode_image_to_base64(image_path: str) -> str:
    """Reads a local image file and converts it into a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_payload(base64_image: str, mime_type: str, prompt_text: str) -> str:
    """Sends a base64 encoded image and prompt to the vision LLM."""
    prompt = prompt_text if prompt_text else "Describe this image in detail."
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            },
        ]
    )
    response = llm_vision.invoke([message])
    return response.content


@app.route("/")
def home():
    """Renders the HTML interface template."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Handles image payload analysis from HTTP POST requests."""
    data = request.get_json()
    image_base64 = data.get("image")
    mime_type = data.get("mime_type", "image/png")
    prompt = data.get("prompt", "Describe this image in detail.")

    if not image_base64:
        return jsonify({"error": "No image payload provided"}), 400

    try:
        result = analyze_image_payload(image_base64, mime_type, prompt)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
