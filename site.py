from flask import Flask, request, jsonify, render_template
import os
import json
import pickle
from rag import (
    generate_enriched_context,
    generate_exam_question_with_ollama,
    generate_exam_response_with_ollama,
    evaluate_student_answer,
)
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


app = Flask(__name__)

# Variables globales pour stocker les composants
rag_components = None
llm_components = None
print("Le site démarre")


def load_rag_system(rag_dir):
    """Charge le système RAG depuis le répertoire"""
    global rag_components

    # Charger les chunks
    with open(os.path.join(rag_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    # Charger l'index FAISS
    index = faiss.read_index(os.path.join(rag_dir, "faiss_index.bin"))

    # Charger les métadonnées
    with open(os.path.join(rag_dir, "metadata_indices.json"), "r") as f:
        serialized_indices = json.load(f)

    # Convertir les listes en ensembles pour une recherche plus rapide
    metadata_indices = {}
    for key, value in serialized_indices.items():
        metadata_indices[key] = {k: set(v) for k, v in value.items()}

    # Charger le modèle d'embeddings
    model = SentenceTransformer(os.path.join(rag_dir, "sentence_transformer_model"))
    print("Modèle d'embeddings chargé :", model)
    print("Test génération embeddings sur un exemple...")
    test_embedding = model.encode(["Test embedding"], show_progress_bar=True)
    print("Embedding de test généré :", test_embedding.shape)
    # Assembler les composants
    rag_components = {
        "index": index,
        "chunks": chunks,
        "model": model,
        "metadata_indices": metadata_indices,
    }

    print("Système RAG chargé avec succès.")


def load_llm_model(model_path):
    """Charge un modèle plus léger si nécessaire"""
    global llm_components

    # Vous pouvez choisir un modèle plus petit
    # Par exemple: "mistralai/Mistral-7B-Instruct-v0.2" au lieu d'un grand modèle

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # S'assurer que pad_token est défini
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Charger le modèle avec des options pour réduire la consommation de mémoire
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    llm_components = (model, tokenizer)
    print("Modèle chargé")


def initialize_app():
    """Initialise les composants au démarrage"""
    # Configurer les chemins
    rag_dir = "test_output"
    model_path = "meta-llama/Llama-3.2-3B"

    # Charger les composants
    load_rag_system(rag_dir)
    load_llm_model(model_path)


initialize_app()


@app.route("/")
def index():
    """Page d'accueil"""
    return render_template("index.html")


@app.route("/api/topics", methods=["GET"])
def get_topics():
    # Extraire les sujets des données
    """Renvoie la liste des sujets disponibles"""
    years = list(rag_components["metadata_indices"]["year"].keys())
    years.sort(reverse=True)

    return jsonify(
        {
            "years": years,
            "parts": ["pt1", "pt2"],
            "content_types": [
                "questions",
                "answers",
                "example_solutions",
                "instructions",
            ],
        }
    )


@app.route("/api/generate_question", methods=["POST"])
def api_generate_question():
    """API pour générer une question et sa réponse modèle"""
    try:
        data = request.json

        topic = data.get("topic", "")
        difficulty = data.get("difficulty", "moyen")
        year = data.get("year", None)
        part = data.get("part", None)

        # Valider les entrées
        if not topic:
            return jsonify({"error": "Le sujet ne peut pas être vide"}), 400

        # Récupérer le contexte RAG
        context = ""
        try:
            context = generate_enriched_context(
                query=f"question {topic} {difficulty}",
                vector_db=rag_components,
                topic=topic,
                content_types=["questions", "example_solutions"],
                top_k=2,
            )
        except Exception as e:
            print(f"Erreur de RAG: {str(e)}")

        # Générer la question
        question_result = generate_exam_question_with_ollama(
            topic=topic,
            difficulty=difficulty,
            context=context,
        )

        # Vérifier si la question a bien été générée
        if not question_result or question_result.get("error", False):
            return jsonify(
                {
                    "question": question_result.get("question", "Erreur de génération"),
                    "error": True,
                }
            ), 200

        # Extraire le texte de la question
        question_text = question_result.get("question", "")

        # Générer la réponse modèle
        model_answer_result = generate_exam_response_with_ollama(
            question=question_text, topic=topic, context=context
        )

        # Combiner les résultats
        combined_result = {
            "question": question_text,
            "topic": topic,
            "difficulty": difficulty,
            "model_answer": model_answer_result.get("model_answer", ""),
            "year_reference": year,
            "part_reference": part,
        }

        # Stocker la réponse modèle en session ou en cache pour une utilisation ultérieure
        # Ici, on pourrait utiliser Flask-Session ou un autre mécanisme de stockage

        return jsonify(combined_result)

    except Exception as e:
        print(f"Erreur API de génération de question: {str(e)}")
        import traceback

        traceback.print_exc()

        return jsonify(
            {"question": f"Une erreur s'est produite: {str(e)}", "error": True}
        ), 200


@app.route("/api/analyse_response", methods=["POST"])
def api_evaluate_answer():
    """API pour analyser la réponse d'un étudiant"""
    try:
        data = request.json

        question = data.get("question", "")
        student_answer = data.get("answer", "")
        model_answer = data.get("model_answer", "")  # Récupérer la réponse modèle

        # Valider les entrées
        if not student_answer:
            return jsonify({"error": "Vous devez donner une réponse"}), 400

        # Si la réponse modèle n'est pas fournie, la générer
        if not model_answer:
            try:
                context = generate_enriched_context(
                    query=question,
                    vector_db=rag_components,
                    content_types=["answers", "example_solutions"],
                    top_k=2,
                )

                model_answer_result = generate_exam_response_with_ollama(
                    question=question,
                    topic="",  # Le sujet peut être extrait de la question
                    context=context,
                )

                model_answer = model_answer_result.get("model_answer", "")
            except Exception as e:
                print(f"Erreur lors de la génération de la réponse modèle: {str(e)}")

        # Évaluer la réponse de l'étudiant
        result = evaluate_student_answer(
            question=question,
            student_answer=student_answer,
            model_answer=model_answer,
            vector_db=rag_components,
        )

        # Vérifier si l'évaluation a bien été générée
        if not result:
            return jsonify({"error": "Résultat d'évaluation vide"}), 500

        # En cas d'erreur dans l'évaluation
        if result.get("error", False):
            return jsonify(
                {
                    "evaluation": result.get("evaluation", "Erreur d'évaluation"),
                    "error": True,
                }
            ), 200

        return jsonify(result)

    except Exception as e:
        print(f"Erreur API d'évaluation de réponse: {str(e)}")
        import traceback

        traceback.print_exc()

        return jsonify(
            {"evaluation": f"Une erreur s'est produite: {str(e)}", "error": True}
        ), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
