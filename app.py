from flask import Flask, request, jsonify
import os
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
        # Forzar que se interprete siempre como JSON
        data = request.get_json(force=True)

        # Debug temporal: imprime lo que llega
        print("📥 RAW request.data:", request.data)
        print("📥 Parsed JSON:", data)

        if not data or "question" not in data:
            return jsonify({
                "error": "Falta el campo 'question'",
                "raw": request.data.decode("utf-8")
            }), 400

        question = data["question"]

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
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)