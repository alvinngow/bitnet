from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import chromadb
import requests
import subprocess
# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Chroma DB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"  # Directory for local data
)
collection = chroma_client.get_or_create_collection("uploaded_documents")

# Endpoint for uploading documents
@app.route('/upload', methods=['POST'])
def upload_documents():
    print('🚀 ~ request.files:', request.files)

    files = request.files
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    try:
        for key in files:
            file = files[key]

            # Save file locally
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Store in Chroma DB
            collection.add(
                documents=[content],
                metadatas=[{"filename": filename}],
                ids=[filename]
            )

            # Trigger Docker container to process the file
            docker_command = [
                "docker", "run", "--rm",
                "-v", f"{os.path.abspath(app.config['UPLOAD_FOLDER'])}:/uploads",
                "bitnet_with_files",
                "python3", "run_inference.py", "-m",
                "Llama3-8B-1.58-100B-tokens-TQ2_0.gguf", "-p",
                f"/uploads/{filename}"
            ]
            subprocess.run(docker_command, check=True)

        return jsonify({"message": "Files uploaded, stored, and processed successfully"}), 200

    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Docker processing failed", "details": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/query', methods=['POST'])
def query_with_bitnet():
    try:
        data = request.json
        user_query = data.get('query')
        if not user_query:
            return jsonify({"error": "Query is required"}), 400

        search_results = collection.query(
            query_texts=[user_query],
            n_results=1 
        )

        if not search_results['documents']:
            return jsonify({"error": "No relevant documents found"}), 404

        relevant_document = search_results['documents'][0][0]  

        bitnet_payload = {
            "context": relevant_document,
            "query": user_query
        }
        bitnet_response = requests.post(BITNET_SERVER_URL, json=bitnet_payload)

        if bitnet_response.status_code != 200:
            return jsonify({
                "error": "Failed to query BitNet",
                "details": bitnet_response.text
            }), 500

        return jsonify({
            "query": user_query,
            "context": relevant_document,
            "response": bitnet_response.json()
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
