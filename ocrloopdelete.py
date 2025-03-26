import requests
import os
import json
import time

OCR_API_KEY = "K83051074288957"
OCR_URL = "https://api.ocr.space/parse/image"

pdf_root_folder = "split_pages"
output_root_folder = "ocr_text"
os.makedirs(output_root_folder, exist_ok=True)

while True:
    # Pour chaque dossier de pdf splité, on applique l'api ocr pour chaque pdf à l'interieur
    for folder in sorted(os.listdir(pdf_root_folder)):
        folder_path = os.path.join(pdf_root_folder, folder)

        if os.path.isdir(folder_path):
            print(f"Traitement du dossier : {folder}")
            output_folder = os.path.join(output_root_folder, folder)
            os.makedirs(output_folder, exist_ok=True)

            for pdf_file in sorted(os.listdir(folder_path)):
                if pdf_file.endswith(".pdf"):
                    file_path = os.path.join(folder_path, pdf_file)
                    print(f"Traitement OCR : {file_path}")

                    with open(file_path, "rb") as f:
                        try:
                            response = requests.post(
                                OCR_URL,
                                files={"file": f},
                                data={
                                    "apikey": OCR_API_KEY,
                                    "language": "eng",
                                    "isOverlayRequired": False,
                                },
                                timeout=60,  # Timeout pour éviter un blocage
                            )
                        except requests.exceptions.RequestException as e:
                            print(f"Erreur réseau : {e}")
                            continue

                    # Vérification de la limite API
                    if (
                        response.status_code == 403
                        and "You may only perform this action" in response.text
                    ):
                        print("Limite API atteinte ! Attente de 3600 secondes...")
                        time.sleep(3600)
                        break  # Stoppe la boucle et reprend l'heure suivante

                    if response.status_code != 200:
                        print(
                            f"Réponse HTTP non valide ({response.status_code}) : {response.text}"
                        )
                        continue

                    if not response.text.strip():
                        print(
                            f"Réponse vide pour {pdf_file}. L'API n'a peut-être pas répondu."
                        )
                        continue

                    try:
                        result = response.json()
                    except json.JSONDecodeError:
                        print(
                            f"Erreur de parsing JSON pour {pdf_file}. Réponse brute : {response.text}"
                        )
                        continue

                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except json.JSONDecodeError:
                            print(
                                f"Impossible de charger le JSON pour {pdf_file}. Réponse brute : {result}"
                            )
                            continue

                    if result and result.get("OCRExitCode") == 1:
                        extracted_text = result["ParsedResults"][0]["ParsedText"]
                        text_file = os.path.join(
                            output_folder, pdf_file.replace(".pdf", ".txt")
                        )
                        with open(text_file, "w", encoding="utf-8") as f:
                            f.write(extracted_text)
                        print(f"Texte extrait et sauvegardé : {text_file}")

                        # Suppression du PDF après traitement
                        os.remove(file_path)
                        print(f"Fichier supprimé : {pdf_file}")
                    else:
                        print(
                            f"Erreur OCR pour {pdf_file} :",
                            result.get("ErrorMessage", "Erreur inconnue"),
                        )

    print("En attente de nouveaux fichiers...")
    time.sleep(10)  # Pause avant de vérifier à nouveau
