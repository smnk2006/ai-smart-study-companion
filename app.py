from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables from .env file

app = Flask(__name__)

# Get your OpenAI API key from environment variable
openai_key = os.getenv("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")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize ChatOpenAI model (reads API key from env internally)
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
)

@app.route("/")
def index():
    return render_template("index.html")  # Make sure you have this template

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
    app.run(debug=True)

