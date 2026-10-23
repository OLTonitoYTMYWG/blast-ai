import os

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MAX_MESSAGES = 20
MAX_MESSAGE_LENGTH = 4000

BLAST_PERSONALITY = """
You are Blast AI, a friendly, creative, helpful AI assistant.

Help users with:
- General questions and conversation.
- Coding and game development.
- Writing and brainstorming.
- Learning and creative projects.

Be honest about your abilities and limitations.
Your name is Blast AI.
"""


@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "name": "Blast AI"
    })


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request."}), 400

    messages = data.get("messages")

    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "Please enter a message."}), 400

    messages = messages[-MAX_MESSAGES:]
    contents = []

    for item in messages:
        if not isinstance(item, dict):
            return jsonify({"error": "Invalid message format."}), 400

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant"):
            return jsonify({"error": "Invalid message role."}), 400

        if not isinstance(content, str) or not content.strip():
            return jsonify({"error": "A message cannot be empty."}), 400

        if len(content) > MAX_MESSAGE_LENGTH:
            return jsonify({"error": "That message is too long."}), 400

        contents.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": content}]
        })

    if contents[-1]["role"] != "user":
        return jsonify({
            "error": "The last message must come from the user."
        }), 400

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("AI_MODEL", "").strip()

    if not api_key:
        return jsonify({
            "error": "GEMINI_API_KEY is not configured."
        }), 503

    if not model:
        return jsonify({
            "error": "AI_MODEL is not configured."
        }), 503

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=BLAST_PERSONALITY,
                max_output_tokens=2048
            )
        )

        reply = response.text

        if not reply:
            return jsonify({
                "error": "The AI returned an empty response."
            }), 502

        return jsonify({"reply": reply})

    except Exception as error:
        app.logger.exception("Blast AI request failed.")
        error_text = str(error).upper()
        error_code = getattr(error, "code", None)

        if (
            error_code == 503
            or "503" in error_text
            or "UNAVAILABLE" in error_text
        ):
            return jsonify({
                "error": "The AI service is temporarily busy. Try again shortly."
            }), 503

        if (
            error_code == 429
            or "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
        ):
            return jsonify({
                "error": "The AI request limit was reached. Please wait and retry."
            }), 429

        if error_code in (401, 403) or "INVALID API KEY" in error_text:
            return jsonify({
                "error": "The API key was rejected. Check your Render settings."
            }), 502

        return jsonify({
            "error": "Blast AI could not get a response. Please try again."
        }), 502


@app.errorhandler(413)
def too_large(_error):
    return jsonify({
        "error": "The request is too large. The limit is 10 MB."
    }), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    app.run(host="0.0.0.0", port=port, debug=False)
