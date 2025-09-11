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