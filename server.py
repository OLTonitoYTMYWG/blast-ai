import os

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024

MAX_MESSAGES = 20
MAX_MESSAGE_LENGTH = 4000

BLAST_PERSONALITY = """
You are Blast AI, an original, friendly AI assistant.

PERSONALITY:
- Be friendly, helpful, creative, and conversational.
- Use appropriate humour and playful banter.
- Give detailed answers when needed.
- Help with coding, game development, writing,
  learning, brainstorming, and general questions.
- Help users create and understand code.
- Respect the user's creative vision.
- Never claim to have done things you haven't done.
- Be honest about uncertainty and limitations.
- Do not claim to have tools you don't have.

IDENTITY:
Your name is Blast AI.
You are your own assistant identity, not Claude or ChatGPT.
"""

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("Gemini API key is not configured.")

    return genai.Client(api_key=api_key)


@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "name": "Blast AI"})


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
            "error": "Please start a new chat."
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
            return jsonify({"error": "A message is too long."}), 400

        contents.append({
            "role": "user" if role == "user" else "model",
            "parts": [{"text": content}]
        })

    if contents[-1]["role"] != "user":
        return jsonify({
            "error": "The last message must come from the user."
        }), 400

    try:
        model = os.environ.get(
            "AI_MODEL", "gemini-3.8-flash"
        )

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
            reply = "I couldn't generate a response. Please try again."

        return jsonify({"reply": reply})

    except Exception:
        app.logger.exception("Blast AI request failed.")
        return jsonify({
            "error": (
                "Blast AI couldn't respond. "
                "Please check the server configuration and try again."
            )
        }), 500


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"error": "The request is too large."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    app.run(host="0.0.0.0", port=port, debug=False)
