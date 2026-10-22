import os

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024

MAX_MESSAGES = 20
MAX_MESSAGE_LENGTH = 4000

BLAST_PERSONALITY = """
You are Blast AI, a friendly, creative, helpful AI assistant.

Help users with:
- General questions and conversation.
- Coding, debugging, and game development.
- Writing, brainstorming, and learning.
- Planning creative projects.

Be honest about your abilities and limitations.
Do not claim to have completed actions you have not done.
Your name is Blast AI.
"""


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    return genai.Client(api_key=api_key)


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

    if len(messages) > MAX_MESSAGES:
        return jsonify({
            "error": "This conversation is too long. Start a new chat."
        }), 400

    contents = []

    for message in messages:
        if not isinstance(message, dict):
            return jsonify({"error": "Invalid message format."}), 400

        role = message.get("role")
        content = message.get("content")

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

    model = os.environ.get("AI_MODEL", "").strip()

    if not model:
        return jsonify({
            "error": "The AI model is not configured in Render."
        }), 503

    try:
        client = get_client()

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=BLAST_PERSONALITY,
                max_output_tokens=4096
            )
        )

        reply = response.text

        if not reply:
            return jsonify({
                "error": "Gemini returned an empty response. Please try again."
            }), 502

        return jsonify({"reply": reply})

    except Exception as error:
        app.logger.exception("Blast AI request failed.")

        error_code = getattr(error, "code", None)
        error_text = str(error).upper()

        if (
            error_code == 503
            or "503" in error_text
            or "UNAVAILABLE" in error_text
        ):
            return jsonify({
                "error": (
                    "Gemini is temporarily busy. "
                    "Please wait a little and try again."
                )
            }), 503

        if (
            error_code == 429
            or "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
        ):
            return jsonify({
                "error": (
                    "Gemini's request limit has been reached. "
                    "Please wait before trying again."
                )
            }), 429

        if error_code in (401, 403) or "INVALID API KEY" in error_text:
            return jsonify({
                "error": "Please check the Gemini API key in Render."
            }), 502

        return jsonify({
            "error": (
                "Blast AI couldn't get a response from Gemini. "
                "Please try again later."
            )
        }), 502


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "The request is too large."
    }), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    app.run(host="0.0.0.0", port=port, debug=False)
