from flask import Flask, request, jsonify
import os
import requests
import tempfile
import mimetypes
import html
import google.generativeai as genai

# Configurer l'API Gemini avec votre clé API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

# Dictionnaire pour stocker les historiques de conversation
sessions = {}

# Taille maximale de l'image (5 Mo)
MAX_IMAGE_SIZE = 5 * 1024 * 1024

def sanitize_input(data):
    """Nettoie les données d'entrée pour éviter les attaques XSS."""
    if isinstance(data, str):
        return html.escape(data.strip())
    return data

def download_image(url):
    """Télécharge une image depuis une URL et retourne le chemin du fichier temporaire."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            content_length = int(response.headers.get('Content-Length', 0))
            if content_length > MAX_IMAGE_SIZE:
                print("Image too large")
                return None
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.flush()
                return temp_file.name
        else:
            print(f"Failed to download image: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def upload_to_gemini(path, mime_type=None):
    """Télécharge le fichier donné sur Gemini."""
    if not mime_type:
        mime_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        return file
    except Exception as e:
        print(f"Erreur lors de l'upload du fichier : {e}")
        return None

# Configuration du modèle avec les paramètres de génération
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

@app.route('/api/gemini', methods=['POST'])
def handle_request():
    try:
        data = request.json
        if not data:
            return jsonify({'message': 'Invalid request, JSON body required'}), 400

        prompt = sanitize_input(data.get('prompt', ''))
        custom_id = sanitize_input(data.get('customId', ''))
        image_url = sanitize_input(data.get('link', ''))

        if not prompt:
            return jsonify({'message': 'Prompt is required'}), 400

        # Récupérer l'historique de la session existante ou en créer une nouvelle
        if custom_id not in sessions:
            sessions[custom_id] = []  # Nouvelle session
        history = sessions[custom_id]

        if image_url:
            # Téléchargement de l'image
            image_path = download_image(image_url)
            if image_path:
                try:
                    file = upload_to_gemini(image_path)
                    if file:
                        history.append({
                            "role": "user",
                            "parts": [file, prompt],
                        })
                    else:
                        return jsonify({'message': 'Failed to upload image to Gemini'}), 500
                finally:
                    # Supprimer le fichier temporaire après utilisation
                    os.remove(image_path)
            else:
                return jsonify({'message': 'Failed to download image or image too large'}), 400
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
        return jsonify({'message': f'Internal Server Error: {str(e)}'}), 500

@app.route('/')
def home():
    return '<h1>Votre API Gemini est en cours d\'exécution...</h1>'

if __name__ == '__main__':
    # Héberger l'application Flask sur 0.0.0.0 pour qu'elle soit accessible publiquement
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
