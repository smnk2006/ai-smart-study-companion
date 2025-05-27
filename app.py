from flask import Flask, render_template, request
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import torch

app = Flask(__name__)

# Load PDFs from data/ folder
loader = DirectoryLoader("./data/", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
)

# Vector store
vector_store = FAISS.from_documents(text_chunks, embeddings)

# Load Ollama model
llm = Ollama(model="llama3")

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Retrieval chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory,
)

# Casual greetings response
simple_responses = {
    "hi": "👋 Hello! How can I help you today?",
    "hello": "Hi there! 😊 Ask me anything from your syllabus or notes!",
    "hey": "Hey! Ready to learn something new?",
    "how are you": "I'm just a bot, but I’m excited to help you with your studies!",
    "thanks": "You're welcome! 😊",
    "thank you": "Always happy to help!",
}

# Check if input is small talk
def is_small_talk(text):
    lower = text.lower().strip()
    for key in simple_responses:
        if key in lower:
            return simple_responses[key]
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    simple_reply = is_small_talk(user_input)

    if simple_reply:
        return simple_reply

    result = chain({"question": user_input})
    answer = result["answer"]

    # Detect intent (brief or detailed)
    user_input_lower = user_input.lower()
    if "8 mark" in user_input_lower or "16 mark" in user_input_lower:
        answer += "\n\n📌 *This is a detailed answer suited for 8 or 16 mark questions.*"
    elif "what is" in user_input_lower or len(user_input_lower.split()) <= 5:
        answer += "\n\n✨ *Let me know if you'd like a more detailed version too!*"

    # Simulate exam relevance (for demo, this is static, you could customize)
    if "ai" in user_input_lower or "artificial intelligence" in user_input_lower:
        answer += "\n\n📝 *This question appeared in the last exam.*"

    # Highlight links if any
    if "http" in answer:
        answer = answer.replace("http", "**http")

    return answer

if __name__ == "__main__":
    app.run(debug=True)
