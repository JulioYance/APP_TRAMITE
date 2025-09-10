import os
from flask import Flask, jsonify
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

app = Flask(__name__)

# RUTA donde se guardará/cargará el índice
INDEX_DIR = "base_vectorial"
INDEX_NAME = "index"

# Embeddings
embeddings = OpenAIEmbeddings()

# Cargar o inicializar FAISS
def load_or_create_faiss():
    index_path = os.path.join(INDEX_DIR, f"{INDEX_NAME}.faiss")

    # Si no existe, crear un FAISS vacío
    if not os.path.exists(index_path):
        os.makedirs(INDEX_DIR, exist_ok=True)
        print("⚠️ No se encontró el índice. Creando uno nuevo...")
        # Creamos un FAISS vacío
        return FAISS.from_texts(["Inicialización del índice"], embeddings)

    # Si existe, cargarlo
    print("✅ Cargando índice existente...")
    return FAISS.load_local(INDEX_DIR, embeddings, index_name=INDEX_NAME, allow_dangerous_deserialization=True)

# Inicializamos el vectorstore
vectorstore = load_or_create_faiss()


@app.route("/")
def home():
    return jsonify({"message": "Servidor Flask con FAISS funcionando 🚀"})


@app.route("/buscar/<texto>")
def buscar(texto):
    """Permite hacer búsqueda en el índice FAISS"""
    docs = vectorstore.similarity_search(texto, k=2)
    resultados = [doc.page_content for doc in docs]
    return jsonify({"query": texto, "resultados": resultados})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)