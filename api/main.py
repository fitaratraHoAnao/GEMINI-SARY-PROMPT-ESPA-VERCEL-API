from flask import Flask, request, jsonify
import os
import google.generativeai as genai

# Configurer l'API Gemini avec votre clé API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

# Dictionnaire pour stocker les historiques de conversation
sessions = {}

def upload_to_gemini(path, mime_type=None):
    """Télécharge le fichier donné sur Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

# Configuration du modèle avec les paramètres de génération
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

@app.route('/', methods=['POST'])
def handle_request():
    data = request.json
    prompt = data.get('prompt', '')  # Question ou prompt de l'utilisateur
    custom_id = data.get('customId', '')  # Identifiant de l'utilisateur ou session
    image_url = data.get('link', '')  # URL de l'image

    # Récupérer l'historique de la session existante ou en créer une nouvelle
    if custom_id not in sessions:
        sessions[custom_id] = []  # Nouvelle session
    history = sessions[custom_id]

    # Ajouter l'image à l'historique si elle est présente
    if image_url:
        file = upload_to_gemini(image_url)
        history.append({
            "role": "user",
            "parts": [file, prompt],
        })
    else:
        history.append({
            "role": "user",
            "parts": [prompt],
        })

    # Démarrer ou continuer une session de chat avec l'historique
    chat_session = model.start_chat(history=history)

    # Envoyer un message dans la session de chat
    response = chat_session.send_message(prompt)

    # Ajouter la réponse du modèle à l'historique
    history.append({
        "role": "model",
        "parts": [response.text],
    })

    # Retourner la réponse du modèle
    return jsonify({'message': response.text})

if __name__ == '__main__':
    # Héberger l'application Flask sur 0.0.0.0 pour qu'elle soit accessible publiquement
    app.run(host='0.0.0.0', port=5000)
