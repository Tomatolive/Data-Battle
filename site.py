from flask import Flask, request, jsonify, render_template
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Variables globales pour stocker les composants
rag_components = None
llm_components = None

def load_rag_system(rag_dir):
    """Charge le système RAG depuis le répertoire"""
    global rag_components
    
    # Charger les chunks
    with open(os.path.join(rag_dir, 'chunks.pkl'), 'rb') as f:
        chunks = pickle.load(f)
    
    # Charger l'index FAISS
    index = faiss.read_index(os.path.join(rag_dir, 'faiss_index.bin'))
    
    # Charger les métadonnées
    with open(os.path.join(rag_dir, 'metadata_indices.json'), 'r') as f:
        serialized_indices = json.load(f)
    
    # Convertir les listes en ensembles pour une recherche plus rapide
    metadata_indices = {}
    for key, value in serialized_indices.items():
        metadata_indices[key] = {k: set(v) for k, v in value.items()}
    
    # Charger le modèle d'embeddings
    model = SentenceTransformer(os.path.join(rag_dir, 'sentence_transformer_model'))
    
    # Assembler les composants
    rag_components = {
        "index": index,
        "chunks": chunks,
        "model": model,
        "metadata_indices": metadata_indices
    }
    
    print("Système RAG chargé avec succès.")

def load_llm_model(model_path):
    """Charge le modèle LLM"""
    global llm_components
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto"
    )
    
    llm_components = (model, tokenizer)
    print("Modèle LLM chargé avec succès.")

def initialize_app():
    """Initialise les composants au démarrage"""
    # Configurer les chemins
    rag_dir = "test_output/chunks.pkl"
    model_path = "path/to/fine_tuned_model"
    
    # Charger les composants
    load_rag_system(rag_dir)
    load_llm_model(model_path)

initialize_app()

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/api/topics', methods=['GET'])
def get_topics():
    # Extraire les sujets des données
    """Renvoie la liste des sujets disponibles"""
    years = list(rag_components["metadata_indices"]["year"].keys())
    years.sort(reverse=True)
    
    return jsonify({
        "years": years,
        "parts": ["pt1", "pt2"],
        "content_types": ["questions", "answers", "example_solutions", "instructions"]
    })

@app.route('/api/generate_question', methods=['POST'])
def api_generate_question():
    """API pour générer une question"""
    data = request.json
    
    topic = data.get('topic', '')
    difficulty = data.get('difficulty', 'moyen')
    year = data.get('year', None)
    part = data.get('part', None)
    
    try:
        # Générer la question
        result = generate_exam_question(
            topic=topic,
            difficulty=difficulty,
            vector_db=rag_components,
            llm_components=llm_components,
            year=year,
            part=part
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate_answer', methods=['POST'])
def api_evaluate_answer():
    """API pour évaluer une réponse"""
    data = request.json
    
    question = data.get('question', '')
    student_answer = data.get('answer', '')
    
    try:
        # Évaluer la réponse
        result = evaluate_student_answer(
            question=question,
            student_answer=student_answer,
            vector_db=rag_components,
            llm_components=llm_components
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def api_search():
    """API pour effectuer une recherche dans les documents"""
    data = request.json
    
    query = data.get('query', '')
    filters = data.get('filters', {})
    top_k = data.get('top_k', 5)
    
    try:
        # Effectuer la recherche
        results = filtered_semantic_search(
            query=query,
            vector_db=rag_components,
            filters=filters,
            top_k=top_k
        )
        
        # Formater les résultats pour l'API
        formatted_results = []
        for result in results:
            chunk = result["chunk"]
            formatted_results.append({
                "text": chunk["text"],
                "year": chunk["year"],
                "part": chunk["part"],
                "content_type": chunk["content_type"],
                "group": chunk["group"],
                "path": chunk["path"],
                "score": result["score"]
            })
        
        return jsonify({"results": formatted_results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
