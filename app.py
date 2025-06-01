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

