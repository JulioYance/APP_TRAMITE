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
        # Intentar parsear como JSON
        data = request.get_json(silent=True)

        # Si falla, intentar manualmente
        if not data:
            try:
                data = json.loads(request.data.decode("utf-8"))
            except:
                data = {}

        # Validar campo 'question'
        if not data or "question" not in data:
            return jsonify({
                "error": "Falta el campo 'question'",
                "raw": request.data.decode("utf-8")
            }), 400

        question = data["question"]

        # Llamar a OpenAI
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
