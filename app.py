from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Cargar la base vectorial
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("base_vectorial", embeddings)

@app.route("/preguntar", methods=["POST"])
def preguntar():
    datos = request.get_json()
    pregunta = datos.get("pregunta", "")
    resultados = vectorstore.similarity_search(pregunta, k=1)
    respuesta = resultados[0].page_content if resultados else "No se encontró información relevante."
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(port=5000)
