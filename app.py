from flask import Flask, request, jsonify, render_template
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env variables like OPENAI_API_KEY

app = Flask(__name__)

# Load OpenAI key securely
openai_key = os.getenv("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")
chat_model = ChatOpenAI(openai_api_key=openai_key, temperature=0.7, model="gpt-3.5-turbo")

@app.route("/")
def index():
    return render_template("index.html")  # Assumes your HTML is saved as 'templates/index.html'

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
