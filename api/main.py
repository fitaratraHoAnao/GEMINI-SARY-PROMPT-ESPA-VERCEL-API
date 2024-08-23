from flask import Flask, request, jsonify
import os
import requests
import tempfile
import google.generativeai as genai

# Configurer l'API Gemini avec votre clé API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

# Dictionnaire pour stocker les historiques de conversation
sessions = {}

def download_image(url):
    """Télécharge une image depuis une URL et retourne le chemin du fichier temporaire."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()
            return temp_file.name
    else:
        return None

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

@app.route('/api/gemini', methods=['POST'])
def handle_request():
    try:
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
            image_path = download_image(image_url)
            if image_path:
                file = upload_to_gemini(image_path)
                if file:
                    history.append({
                        "role": "user",
                        "parts": [file, prompt],
                    })
                else:
                    return jsonify({'message': 'Failed to upload image to Gemini'}), 500
            else:
                return jsonify({'message': 'Failed to download image'}), 500
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

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'message': 'Internal Server Error'}), 500

if __name__ == '__main__':
    # Héberger l'application Flask sur 0.0.0.0 pour qu'elle soit accessible publiquement
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
