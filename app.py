import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# HuggingFace imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Load environment variables
load_dotenv()

# --- 1. Load and Process Documents ---
def process_documents(directory="docs"):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

# --- 2. Create Vector Database ---
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# --- 3. Set Up RAG Chain ---
def get_qa_chain(vector_db):
    # Use local Flan-T5 via HuggingFace pipeline
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=0  # set -1 for CPU
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )
    return qa_chain

# --- Flask App ---
app = Flask(__name__)
qa_chain = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    global qa_chain
    if qa_chain is None:
        return jsonify({"answer": "Chatbot not ready. Please wait."})

    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({"answer": "Please enter a question."})

    try:
        result = qa_chain.invoke({"query": question})
        answer = result.get("result") if isinstance(result, dict) else result
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"An error occurred: {str(e)}"})

# --- Main ---
if __name__ == "__main__":
    if not os.path.exists("docs"):
        os.makedirs("docs")
        print("Please place your Verité PDF files in the 'docs' directory.")
    else:
        print("Processing Verité documents...")
        chunks = process_documents()
        if chunks:
            vector_db = create_vector_db(chunks)
            qa_chain = get_qa_chain(vector_db)
            print("Chatbot is ready. You can now run the app.")
        else:
            print("No PDF files found in the 'docs' directory. Chatbot will not be ready.")

    app.run(debug=True)
 