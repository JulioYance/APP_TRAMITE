import os
from flask import Flask, jsonify
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

app = Flask(__name__)

# Configuración
INDEX_DIR = "base_vectorial"
INDEX_NAME = "index"
INDEX_FILE = os.path.join(INDEX_DIR, f"{INDEX_NAME}.faiss")

# Embeddings
embeddings = OpenAIEmbeddings()

# Función robusta para cargar o crear FAISS
def load_or_create_faiss():
    try:
        # Si no existe o está vacío, creamos uno nuevo
        if not os.path.exists(INDEX_FILE) or os.path.getsize(INDEX_FILE) == 0:
            os.makedirs(INDEX_DIR, exist_ok=True)
            print("⚠️ Índice no encontrado o vacío. Creando uno nuevo...")
            faiss_index = FAISS.from_texts(["Índice inicializado"], embeddings)
            faiss_index.save_local(INDEX_DIR, index_name=INDEX_NAME)
            return faiss_index

        # Intentamos cargar el índice existente
        print("✅ Cargando índice existente...")
        return FAISS.load_local(INDEX_DIR, embeddings, index_name=INDEX_NAME, allow_dangerous_deserialization=True)

    except Exception as e:
        print(f"❌ Error cargando índice: {e}")
        print("🔄 Creando índice nuevo...")
        faiss_index = FAISS.from_texts(["Índice reinicializado"], embeddings)
        faiss_index.save_local(INDEX_DIR, index_name=INDEX_NAME)
        return faiss_index


# Inicializamos FAISS
vectorstore = load_or_create_faiss()


@app.route("/")
def home():
    return jsonify({"message": "Servidor Flask con FAISS funcionando 🚀"})


@app.route("/buscar/<texto>")
def buscar(texto):
    docs = vectorstore.similarity_search(texto, k=2)
    resultados = [doc.page_content for doc in docs]
    return jsonify({"query": texto, "resultados": resultados})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)