from flask import Flask, request, jsonify
import os
import json
from openai import OpenAI

app = Flask(__name__)

# Leer la API key desde Railway
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ No se encontró la variable OPENAI_API_KEY en Railway")

client = OpenAI(api_key=api_key)

@app.route("/")
def home():
    return "✅ API funcionando en Railway"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Manejo robusto de encoding (UTF-8 y Latin-1)
        try:
            raw_text = request.data.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = request.data.decode("latin-1")

        try:
            data = json.loads(raw_text)
        except Exception:
            return jsonify({"error": "El cuerpo no es un JSON válido"}), 400

        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Falta el campo 'question'"}), 400

        # Intentar respuesta real con OpenAI
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres un asistente útil."},
                    {"role": "user", "content": question},
                ]
            )
            answer = response.choices[0].message.content
            return jsonify({"answer": answer})

        except Exception as e:
            # Si es error de cuota, devolver respuesta simulada
            if "insufficient_quota" in str(e):
                return jsonify({
                    "answer": f"(Respuesta simulada) Recibí tu pregunta: '{question}'. Tu API Key no tiene crédito disponible."
                })
            # Otros errores
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
from flask import Flask, request, jsonify
import os, json
from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from procesar_documentos import procesar_documentos

app = Flask(__name__)

# ========================
# Configuración
# ========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ No se encontró OPENAI_API_KEY en Railway")

client = OpenAI(api_key=api_key)

CARPETA_INDEX = "faiss_index"
CARPETA_DOCS = "documentos"

# ========================
# Cargar o crear índice
# ========================
def load_or_create_faiss():
    if os.path.exists(CARPETA_INDEX):
        try:
            embeddings = OpenAIEmbeddings(api_key=api_key)
            return FAISS.load_local(CARPETA_INDEX, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ Error cargando índice: {e}. Creando uno nuevo...")
    return procesar_documentos(api_key, CARPETA_DOCS, CARPETA_INDEX)

vectorstore = load_or_create_faiss()

# ========================
# Rutas
# ========================
@app.route("/")
def home():
    return "✅ API RAG funcionando en Railway"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Decodificar request en JSON
        try:
            raw_text = request.data.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = request.data.decode("latin-1")
        data = json.loads(raw_text)

        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Falta el campo 'question'"}), 400

        # Buscar contexto en FAISS
        docs = vectorstore.similarity_search(question, k=3)
        contexto = "\n".join([d.page_content for d in docs])

        # Enviar a GPT
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Responde SOLO en base a los documentos cargados."},
                {"role": "user", "content": f"Pregunta: {question}\n\nContexto:\n{contexto}"}
            ]
        )
        answer = response.choices[0].message.content

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
