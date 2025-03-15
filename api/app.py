
from flask import Flask, request, jsonify
import os
import requests
import tempfile
import google.generativeai as genai
import PyPDF2
import mimetypes

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

# Dictionary to store conversation histories
sessions = {}

def download_file(url):
    """Download a file from URL and return the temporary file path."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        content_type = response.headers.get('content-type')
        extension = mimetypes.guess_extension(content_type) or '.tmp'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file.flush()
            return temp_file.name, content_type
    return None, None

def extract_pdf_text(pdf_path):
    """Extract text from PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def upload_to_gemini(path, mime_type=None):
    """Upload file to Gemini."""
    return genai.upload_file(path, mime_type=mime_type)

# Generation config
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
        prompt = data.get('prompt', '')
        custom_id = data.get('customId', '')
        file_urls = data.get('links', [])  # Now accepting an array of URLs
        
        if not isinstance(file_urls, list):
            file_urls = [file_urls] if file_urls else []

        if custom_id not in sessions:
            sessions[custom_id] = []
        history = sessions[custom_id]

        # Process all files
        files_content = []
        for url in file_urls:
            file_path, mime_type = download_file(url)
            if file_path:
                if mime_type and 'pdf' in mime_type:
                    # For PDFs, extract text and add it to prompt
                    pdf_text = extract_pdf_text(file_path)
                    prompt = f"Here's the content of the PDF:\n{pdf_text}\n\nUser question: {prompt}"
                else:
                    # For images, upload to Gemini
                    file = upload_to_gemini(file_path, mime_type=mime_type)
                    if file:
                        files_content.append(file)
                os.unlink(file_path)  # Clean up temporary file

        # Create message parts
        message_parts = files_content + [prompt] if files_content else [prompt]
        
        # Add to history
        history.append({
            "role": "user",
            "parts": message_parts,
        })

        # Start or continue chat session
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(prompt)

        # Add model response to history
        history.append({
            "role": "model",
            "parts": [response.text],
        })

        return jsonify({'message': response.text})

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'message': 'Internal Server Error'}), 500

@app.route('/')
def home():
    return '<h1>Your Gemini API is running...</h1>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
