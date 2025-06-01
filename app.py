from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables from .env (optional on Render, since it's set in dashboard)
load_dotenv()

app = Flask(__name__)

# Correct way: get API key from environment
openai_key = os.getenv("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")

# Safety check
if not openai_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")

# Set up ChatOpenAI with key
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai_key
)

@app.route("/")
def index():
    return render_template("index.html")  # Ensure templates/index.html exists

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input", "")
    if not user_input:
        return jsonify({"answer": "Please enter a message."})

    try:
        messages = [HumanMessage(content=user_input)]
        response = chat_model(messages)
        return jsonify({"answer": response.content})
    except Exception as e:
        return jsonify({"answer": f"❌ Error: {str(e)}"})

if __name__ == "__main__":
    # Required for Render (host=0.0.0.0 and port from environment)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
