import os

from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

app = Flask(__name__, static_folder="static")

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024

MAX_MESSAGES = 20
MAX_MESSAGE_LENGTH = 4000

BLAST_PERSONALITY = """
You are Blast AI, an original, friendly AI assistant.

PERSONALITY:
- Be friendly, helpful, creative, and conversational.
- Enjoy appropriate humour, playful banter, and jokes.
- Be capable of serious, detailed answers when needed.
- Help with programming, game development, writing,
  learning, brainstorming, and general questions.
- Adapt your response length to the user's needs.
- Respect the user's ideas and creative vision.
- Never claim to have performed actions you have not performed.
- Be honest about uncertainty and your limitations.
- Do not claim to remember previous conversations unless
  their relevant information is actually available to you.
- Do not claim to have internet access or tools that are
  not provided to you.

IDENTITY:
Your name is Blast AI.
You are your own assistant identity, not Claude or ChatGPT.
You can discuss other AI assistants honestly and respectfully.
"""

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("The AI API key has not been configured.")

    return OpenAI(api_key=api_key)


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
            "error": "This conversation is too long for one request. "
                     "Please start a new chat."
        }), 400

    clean_messages = []

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

        clean_messages.append({
            "role": role,
            "content": content
        })

    if clean_messages[-1]["role"] != "user":
        return jsonify({
            "error": "The last message must come from the user."
        }), 400

    model = os.environ.get("AI_MODEL")

    if not model:
        return jsonify({
            "error": "Blast AI's model has not been configured yet."
        }), 503

    try:
        client = get_client()

        response = client.responses.create(
            model=model,
            instructions=BLAST_PERSONALITY,
            input=clean_messages
        )

        reply = response.output_text

        if not reply:
            reply = "I couldn't generate a response. Please try again."

        return jsonify({"reply": reply})

    except Exception:
        app.logger.exception("Blast AI request failed.")

        return jsonify({
            "error": (
                "Blast AI couldn't respond. Please try again. "
                "If the problem continues, the server configuration "
                "may need checking."
            )
        }), 500


@app.errorhandler(413)
def request_too_large(error):
    return jsonify({"error": "The request is too large."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
      )
          
