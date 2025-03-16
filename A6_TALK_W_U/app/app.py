from flask import Flask, request, jsonify, render_template
import os
import torch

from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set the default directory to the current directory ***
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Load embedding model
model_name = 'hkunlp/instructor-base'
embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

# Load vector store using FAISS
vector_path = 'vector-store'
db_file_name = 'nlp_vector_store'
vector_store = FAISS.load_local(
    folder_path=os.path.join(vector_path, db_file_name),
    embeddings=embedding_model,
    index_name='nlp',
    allow_dangerous_deserialization=True
)

# Load TinyLlama model using HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id("TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation", device=0)

# Define retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Create QA prompt template
prompt_template = """Answer the following question based solely on the provided context. 
If the answer is not present in the context, say 'I don't know.'

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Setup RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type='stuff',
    return_source_documents=True,  # Return source docs for additional context
    chain_type_kwargs={"prompt": prompt}
)

@app.route("/")
def index():
    """Render chatbot UI"""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def get_answer():
    """Handles user queries and retrieves responses from the LLM"""
    data = request.json
    query = data.get("question", "")
    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Use the correct key "query" for the QA chain
        result = qa_chain.invoke({"query": query}) 
        match = re.search(r'Answer:\s*(.+)', result["result"])
        first_answer = match.group(1).strip() if match else "No answer found."

        source_documents = response.get("source_documents", [])
        sources = [doc.metadata.get("source", "Unknown") for doc in source_documents]

        return jsonify({
            "question": query,
            "answer": first_answer,
            "sources": sources
        })
    except Exception as e:
        # Return JSON error message rather than HTML error page
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
