import os
import json
import re
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any, Union, Tuple
import ollama
import requests


#################################
#   Chargeur de données adapté  #
#################################
def load_text_documents(root_dir: str) -> List[Dict[str, Any]]:
    """
    Charge tous les fichiers texte de la structure organisée en un format adapté au RAG

    Args:
        root_dir: Chemin racine des fichiers texte extraits

    Returns:
        Liste de documents avec métadonnées
    """
    documents = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(dirpath, filename)

            # Extraire les métadonnées du chemin
            rel_path = os.path.relpath(file_path, root_dir)
            parts = rel_path.split(os.sep)

            if len(parts) < 2:
                continue

            # Structure: année_type_partie_langue/groupe.txt
            exam_info = parts[0].split("_")

            if len(parts) >= 2 and len(exam_info) >= 3:
                try:
                    # Lire le contenu du fichier
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()

                    # Extraire l'année (première partie avant underscore)
                    year = exam_info[0] if exam_info[0].isdigit() else None

                    # Déterminer le type de contenu
                    content_type = None
                    if "answers" in parts[0]:
                        content_type = "answers"
                    elif "ex-rep" in parts[0]:
                        content_type = "example_solutions"
                    elif "instructions" in parts[0]:
                        content_type = "instructions"
                    else:
                        content_type = "questions"

                    # Déterminer la partie
                    part = None
                    for part_id in ["pt1", "pt2"]:
                        if part_id in parts[0]:
                            part = part_id
                            break

                    # Extraire le numéro de groupe
                    group = None
                    if "group_" in parts[1]:
                        group = int(parts[1].replace("group_", "").replace(".txt", ""))

                    # Créer le document avec métadonnées
                    document = {
                        "content": content,
                        "path": rel_path,
                        "year": year,
                        "part": part,
                        "content_type": content_type,
                        "group": group,
                        "filename": filename,
                    }

                    documents.append(document)

                except Exception as e:
                    print(f"Erreur lors du traitement de {file_path}: {e}")

    print(f"Chargé {len(documents)} documents au total.")
    return documents


#########################################################
#   Organisation des chunks avec métadonnées enrichies  #
#########################################################
def create_structured_chunks(
    documents: List[Dict[str, Any]],
    max_chunk_size: int = 1000,
    min_chunk_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Crée des chunks en conservant la structure et les métadonnées des documents

    Args:
        documents: Liste des documents chargés
        max_chunk_size: Taille maximale d'un chunk en caractères
        min_chunk_size: Taille minimale d'un chunk en caractères

    Returns:
        Liste de chunks avec métadonnées
    """
    chunks = []

    for doc in documents:
        content = doc["content"]

        # Pour les documents courts, les garder intacts
        if len(content) <= max_chunk_size:
            chunk = {
                "text": content,
                "year": doc["year"],
                "part": doc["part"],
                "content_type": doc["content_type"],
                "group": doc["group"],
                "path": doc["path"],
                "is_complete_document": True,
            }
            chunks.append(chunk)
            continue

        # Pour les documents plus longs, diviser en paragraphes
        paragraphs = re.split(r"\n\s*\n", content)

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Enregistrer le chunk actuel s'il est assez grand
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(
                        {
                            "text": current_chunk.strip(),
                            "year": doc["year"],
                            "part": doc["part"],
                            "content_type": doc["content_type"],
                            "group": doc["group"],
                            "path": doc["path"],
                            "is_complete_document": False,
                        }
                    )

                # Commencer un nouveau chunk
                current_chunk = para + "\n\n"

        # Ajouter le dernier chunk s'il est assez grand
        if len(current_chunk) >= min_chunk_size:
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "year": doc["year"],
                    "part": doc["part"],
                    "content_type": doc["content_type"],
                    "group": doc["group"],
                    "path": doc["path"],
                    "is_complete_document": False,
                }
            )

    return chunks


#################################################################################
#   Enrichissement des métadonnées avec des liens entre questions et réponses   #
#################################################################################
def link_questions_with_answers(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrichit les chunks en créant des références entre questions et réponses

    Args:
        chunks: Liste des chunks créés

    Returns:
        Chunks enrichis avec des références croisées
    """
    # Créer des dictionnaires pour accéder rapidement aux chunks par année, partie et groupe
    questions_dict = {}
    answers_dict = {}
    solutions_dict = {}

    for i, chunk in enumerate(chunks):
        key = f"{chunk['year']}_{chunk['part']}_{chunk['group']}"

        if chunk["content_type"] == "questions":
            questions_dict[key] = i
        elif chunk["content_type"] == "answers":
            answers_dict[key] = i
        elif chunk["content_type"] == "example_solutions":
            solutions_dict[key] = i

    # Enrichir les chunks avec des références
    for i, chunk in enumerate(chunks):
        key = f"{chunk['year']}_{chunk['part']}_{chunk['group']}"

        # Pour les questions, ajouter les références aux réponses et solutions
        if chunk["content_type"] == "questions":
            if key in answers_dict:
                chunks[i]["answer_ref"] = answers_dict[key]
            if key in solutions_dict:
                chunks[i]["solution_ref"] = solutions_dict[key]

        # Pour les réponses, ajouter les références aux questions et solutions
        elif chunk["content_type"] == "answers":
            if key in questions_dict:
                chunks[i]["question_ref"] = questions_dict[key]
            if key in solutions_dict:
                chunks[i]["solution_ref"] = solutions_dict[key]

        # Pour les solutions, ajouter les références aux questions et réponses
        elif chunk["content_type"] == "example_solutions":
            if key in questions_dict:
                chunks[i]["question_ref"] = questions_dict[key]
            if key in answers_dict:
                chunks[i]["answer_ref"] = answers_dict[key]

    return chunks


#########################################################################
#   Création d'une base de connaissances avec métadonnées filtrables    #
#########################################################################
def create_advanced_vector_db(
    chunks: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2"
):
    """
    Crée une base de connaissances vectorielle avancée avec filtres par métadonnées

    Args:
        chunks: Liste des chunks enrichis
        model_name: Nom du modèle d'embeddings

    Returns:
        Composants du système RAG
    """
    # Charger le modèle d'embeddings
    model = SentenceTransformer(model_name)

    # Extraire les textes pour l'embedding
    texts = [chunk["text"] for chunk in chunks]

    # Générer les embeddings
    print("Génération des embeddings...\n")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    # Construire l'index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    # Créer des indices pour les métadonnées pour faciliter le filtrage
    metadata_indices = {"year": {}, "part": {}, "content_type": {}, "group": {}}

    for i, chunk in enumerate(chunks):
        # Indexer par année
        year = chunk["year"]
        if year not in metadata_indices["year"]:
            metadata_indices["year"][year] = []
        metadata_indices["year"][year].append(i)

        # Indexer par partie
        part = chunk["part"]
        if part not in metadata_indices["part"]:
            metadata_indices["part"][part] = []
        metadata_indices["part"][part].append(i)

        # Indexer par type de contenu
        content_type = chunk["content_type"]
        if content_type not in metadata_indices["content_type"]:
            metadata_indices["content_type"][content_type] = []
        metadata_indices["content_type"][content_type].append(i)

        # Indexer par groupe
        group = chunk["group"]
        if group not in metadata_indices["group"]:
            metadata_indices["group"][group] = []
        metadata_indices["group"][group].append(i)

    return {
        "index": index,
        "chunks": chunks,
        "model": model,
        "metadata_indices": metadata_indices,
    }


#################################################
#   Fonction de recherche avancée avec filtres  #
#################################################
def filtered_semantic_search(
    query: str,
    vector_db: Dict[str, Any],
    filters: Dict[str, Any] = None,
    top_k: int = 5,
):
    """
    Effectue une recherche sémantique avec filtres sur les métadonnées

    Args:
        query: Requête de recherche
        vector_db: Base de connaissances vectorielle
        filters: Filtres sur les métadonnées (année, partie, type, groupe)
        top_k: Nombre de résultats à retourner

    Returns:
        Liste des chunks les plus pertinents
    """
    # Créer l'embedding de la requête
    query_embedding = (
        vector_db["model"].encode([query])[0].reshape(1, -1).astype("float32")
    )
    distances, indices = vector_db["index"].search(query_embedding, top_k)
    # Si aucun filtre n'est spécifié, effectuer une recherche globale
    if filters is None or not filters:
        distances, indices = vector_db["index"].search(query_embedding, top_k)
        results = []

        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(vector_db["chunks"]):
                continue

            chunk = vector_db["chunks"][idx]
            results.append({"chunk": chunk, "score": float(distances[0][i])})

        return results

    # Si des filtres sont spécifiés, rechercher dans un sous-ensemble
    candidate_indices = set()
    first_filter = True

    # Appliquer les filtres
    for filter_key, filter_value in filters.items():
        if (
            filter_key not in vector_db["metadata_indices"]
            or filter_value not in vector_db["metadata_indices"][filter_key]
        ):
            # Si le filtre ne correspond à aucun document, renvoyer un résultat vide
            return []

        # Récupérer les indices correspondant au filtre
        filtered_indices = set(vector_db["metadata_indices"][filter_key][filter_value])

        # Intersection avec les résultats précédents ou initialisation
        if first_filter:
            candidate_indices = filtered_indices
            first_filter = False
        else:
            candidate_indices &= filtered_indices

    # Si aucun document ne correspond à tous les filtres, renvoyer un résultat vide
    if not candidate_indices:
        return []

    # Convertir en liste pour l'indexation
    candidate_list = list(candidate_indices)

    # Créer un sous-index temporaire pour la recherche
    sub_embeddings = np.array(
        [vector_db["index"].reconstruct(i) for i in candidate_list]
    ).astype("float32")
    sub_index = faiss.IndexFlatL2(sub_embeddings.shape[1])
    sub_index.add(sub_embeddings)

    # Effectuer la recherche dans le sous-index
    distances, sub_indices = sub_index.search(
        query_embedding, min(top_k, len(candidate_list))
    )

    # Récupérer les résultats
    results = []
    for i, sub_idx in enumerate(sub_indices[0]):
        if sub_idx < 0 or sub_idx >= len(candidate_list):
            continue

        idx = candidate_list[sub_idx]
        chunk = vector_db["chunks"][idx]

        results.append({"chunk": chunk, "score": float(distances[0][i])})

    return results


#####################################################################
#   Génération de contexte enrichi pour des questions spécifiques   #
#####################################################################
def generate_enriched_context(
    query: str,
    vector_db: Dict[str, Any],
    topic: str = None,
    year: str = None,
    part: str = None,
    content_types: List[str] = None,
    top_k: int = 5,
):
    """
    Génère un contexte enrichi pour une requête spécifique

    Args:
        query: Requête de recherche
        vector_db: Base de connaissances vectorielle
        topic: Sujet spécifique à chercher
        year: Année spécifique
        part: Partie spécifique
        content_types: Types de contenu à inclure
        top_k: Nombre de résultats par catégorie

    Returns:
        Contexte formaté
    """
    if content_types is None:
        content_types = ["questions", "answers", "example_solutions"]

    # Construire les filtres de base
    base_filters = {}
    if year is not None:
        base_filters["year"] = year
    if part is not None:
        base_filters["part"] = part

    # Enrichir la requête avec le sujet si spécifié
    enhanced_query = query
    if topic is not None:
        enhanced_query = f"{topic} {query}"

    all_results = []

    # Effectuer une recherche pour chaque type de contenu
    for content_type in content_types:
        # Ajouter le filtre de type de contenu
        filters = base_filters.copy()
        filters["content_type"] = content_type

        # Effectuer la recherche
        results = filtered_semantic_search(enhanced_query, vector_db, filters, top_k)
        all_results.extend(results)

    # Trier par score
    all_results.sort(key=lambda x: x["score"])

    # Construire le contexte enrichi
    context_parts = []

    # Fonction pour formater un chunk
    def format_chunk(chunk, prefix=""):
        source_info = f"[Année: {chunk['year']}, Partie: {chunk['part']}, "
        source_info += f"Groupe: {chunk['group']}, Type: {chunk['content_type']}]"

        return f"{prefix}{source_info}\n\n{chunk['text']}"

    # Ajouter les résultats au contexte
    for result in all_results[:top_k]:
        chunk = result["chunk"]

        # Formater le chunk principal
        context_parts.append(format_chunk(chunk))

        # Ajouter les références liées (questions, réponses, solutions)
        if chunk["content_type"] == "questions" and "answer_ref" in chunk:
            answer_chunk = vector_db["chunks"][chunk["answer_ref"]]
            context_parts.append(
                format_chunk(answer_chunk, prefix="RÉPONSE ASSOCIÉE: ")
            )

            if "solution_ref" in chunk:
                solution_chunk = vector_db["chunks"][chunk["solution_ref"]]
                context_parts.append(
                    format_chunk(solution_chunk, prefix="SOLUTION ASSOCIÉE: ")
                )

        elif chunk["content_type"] == "answers" and "question_ref" in chunk:
            question_chunk = vector_db["chunks"][chunk["question_ref"]]
            context_parts.append(
                format_chunk(question_chunk, prefix="QUESTION ASSOCIÉE: ")
            )

    # Joindre les parties avec des séparateurs
    formatted_context = "\n\n" + "=" * 50 + "\n\n".join(context_parts)

    return formatted_context


########################################################
#   Script principal pour construire le système RAG    #
########################################################
def build_exam_rag_system(text_root_dir: str, output_dir: str):
    """
    Construit un système RAG complet adapté aux examens d'ingénieur brevet

    Args:
        text_root_dir: Répertoire racine des fichiers texte
        output_dir: Répertoire de sortie pour les composants RAG
    """
    print("Étape 1: Chargement des documents texte...")
    documents = load_text_documents(text_root_dir)

    print("Étape 2: Création de chunks structurés...")
    chunks = create_structured_chunks(documents)

    print("Étape 3: Enrichissement avec liens entre questions et réponses...")
    enriched_chunks = link_questions_with_answers(chunks)

    print("Étape 4: Création de la base de connaissances vectorielle...")
    vector_db = create_advanced_vector_db(enriched_chunks)

    print("Étape 5: Sauvegarde des composants...")
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder les chunks
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(enriched_chunks, f)

    # Sauvegarder l'index FAISS
    faiss.write_index(vector_db["index"], os.path.join(output_dir, "faiss_index.bin"))

    # Sauvegarder les métadonnées
    with open(os.path.join(output_dir, "metadata_indices.json"), "w") as f:
        # Convertir les ensembles en listes pour la sérialisation JSON
        serializable_indices = {}
        for key, value in vector_db["metadata_indices"].items():
            serializable_indices[key] = {k: list(v) for k, v in value.items()}
        json.dump(serializable_indices, f)

    # Sauvegarder le modèle
    vector_db["model"].save(os.path.join(output_dir, "sentence_transformer_model"))

    print(f"Système RAG sauvegardé dans {output_dir}")


##################################################
# Génération de questions d'examen avec Ollama   #
##################################################
def generate_exam_question_with_ollama(topic, difficulty, context=""):
    print("Passage par generate_exam_question_with_ollama")
    """Génère une question d'examen en utilisant Ollama avec la bonne syntaxe API"""

    # Choisir un bon modèle
    model_name = "llama3"  # ou "mistral", "phi3", etc.

    # Créer un prompt bien structuré
    prompt = f"""
Crée une question d'examen sur le sujet: {topic}
Niveau: {difficulty}

Format requis:

QUESTION:
[Ta question ici]

SOLUTION:
[Ta solution détaillée]

CRITÈRES D'ÉVALUATION:
[Les critères]

ERREURS COURANTES:
[Les erreurs typiques]

Contexte additionnel pour t'aider:
{context}
"""

    try:
        # Appeler Ollama avec la syntaxe correcte
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            },
        )
        # Vérifier le statut de la réponse
        if response.status_code == 200:
            try:
                result = response.json()
                generated_text = result.get("response", "")
            except json.JSONDecodeError:
                # Si nous avons toujours une erreur de décodage JSON, traiter comme stream
                text_response = response.text
                generated_text = ""

                # Traiter chaque ligne JSON séparément
                for line in text_response.strip().split("\n"):
                    try:
                        line_data = json.loads(line)
                        if "response" in line_data:
                            generated_text += line_data["response"]
                    except json.JSONDecodeError:
                        continue
        else:
            generated_text = f"Erreur API: {response.status_code}"

        return {
            "question": generated_text,
            "topic": topic,
            "difficulty": difficulty,
            "source": "ollama",
        }

    except Exception as e:
        print(f"Erreur avec Ollama: {str(e)}")
        return {
            "question": f"Erreur: {str(e)}",
            "topic": topic,
            "difficulty": difficulty,
            "error": True,
        }


#################################################
#   Intégration pour l'évaluation de réponses   #
#################################################
def evaluate_student_answer(
    question: str, student_answer: str, vector_db: Dict[str, Any], llm_components: tuple
):
    """
    Évalue la réponse d'un étudiant à une question

    Args:
        question: Question posée
        student_answer: Réponse de l'étudiant
        vector_db: Base de connaissances vectorielle
        llm_components: Modèle LLM et tokenizer

    Returns:
        Évaluation de la réponse
    """
    llm, tokenizer = llm_components

    # Récupérer le contexte pertinent
    context = generate_enriched_context(
        query=question[:100],  # Utiliser le début de la question comme requête
        vector_db=vector_db,
        content_types=["answers", "example_solutions"],
        top_k=3,
    )

    # Construire le prompt
    system_prompt = f"""
    Tu es un expert en évaluation pour le concours d'ingénieur brevet européen.
    Ta tâche est d'évaluer la réponse d'un étudiant à la question suivante:
    
    Question:
    {question}
    
    Réponse de l'étudiant:
    {student_answer}
    
    Utilise les exemples et solutions ci-dessous pour évaluer la réponse:
    {context}
    
    Ton évaluation doit inclure:
    1. Un score sur 10
    2. Une analyse détaillée des points forts et des points faibles
    3. Des suggestions d'amélioration spécifiques
    4. Une comparaison avec les solutions de référence
    
    Format de sortie:
    SCORE: [/10]
    
    ÉVALUATION DÉTAILLÉE:
    [Ton analyse]
    
    POINTS FORTS:
    [Liste des points forts]
    
    POINTS À AMÉLIORER:
    [Liste des points à améliorer]
    
    COMPARAISON AVEC LA RÉFÉRENCE:
    [Comparaison avec les réponses modèles]
    
    CONSEILS:
    [Conseils pour améliorer]
    """

    # Générer l'évaluation
    inputs = tokenizer(system_prompt, return_tensors="pt").to(llm.device)

    with torch.no_grad():
        output = llm.generate(
            inputs.input_ids,
            max_new_tokens=1024,
            temperature=0.3,  # Température plus basse pour l'évaluation
            top_p=0.9,
            do_sample=True,
        )

    # Décoder la sortie
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)
    response = full_output[len(system_prompt) :].strip()

    # Extraire le score (si présent)
    score = None
    if "SCORE:" in response:
        score_line = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*10", response)
        if score_line:
            try:
                score = float(score_line.group(1))
            except:
                pass

    # Formater la réponse
    formatted_response = {
        "evaluation": response,
        "score": score,
        "question": question,
        "student_answer": student_answer,
    }

    return formatted_response


if __name__ == "__main__":
    build_exam_rag_system("ocr_text", "test_output")
