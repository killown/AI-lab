# Multimodal Image Inspector

![Multimodal Image Inspector Demo](multimodal-image-inspector/demo.gif)

A lightweight, local web interface and Flask API powered by LangChain and Qwen 2.5 VL (qwen2.5vl:7b) via Ollama. It allows users to upload local image files and analyze them using visual-language queries.

## Features

- Local Execution: Operates entirely offline using local Ollama models.
- Multimodal Analysis: Powered by qwen2.5vl:7b for high-accuracy document parsing, OCR, and visual descriptions.
- Flask Web Interface: Clean UI to upload images and enter custom prompts.
- REST API: Simple /analyze JSON endpoint for programmatic integration.

## Project Structure

multimodal-image-inspector/
├── app.py
├── templates/
│   └── index.html
└── README.md

## Prerequisites

1. Ollama installed and running locally.
2. Pull the Qwen 2.5 VL model:
   ollama pull qwen2.5vl:7b
3. Python 3.10+ and required packages:
   pip install flask langchain-core langchain-ollama

## Getting Started

1. Navigate to the project folder:
   cd multimodal-image-inspector

2. Start the Flask application:
   python app.py

3. Open your browser at:
   http://127.0.0.1:5000

## API Usage

Endpoint: POST /analyze

Request Body:
{
  "image": "<base64_encoded_string>",
  "mime_type": "image/png",
  "prompt": "Describe this image in detail."
}

Response Body:
{
  "result": "Model visual analysis here..."
}
