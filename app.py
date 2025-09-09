from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

app = Flask(__name__)

# Cargar la base vectorial
vectorstore = FAISS.load_local("base_vectorial", OpenAIEmbeddings())

@app.route("/preguntar", methods=["POST"])
def preguntar():
    datos = request.get_json()
    pregunta = datos.get("pregunta", "")
    resultados = vectorstore.similarity_search(pregunta, k=1)
    respuesta = resultados[0].page_content if resultados else "No se encontró información relevante."
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(port=5000)
