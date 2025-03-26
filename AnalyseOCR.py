import requests
import os
import json

OCR_API_KEY = "K83051074288957"

pdf_root_folder = "split_pages"

output_root_folder = "ocr_text"
os.makedirs(output_root_folder, exist_ok=True)

OCR_URL = "https://api.ocr.space/parse/image"
# Pour chaque dossier de pdf splité, on applique l'api ocr pour chaque pdf à l'interieur
for folder in sorted(os.listdir(pdf_root_folder)):
    folder_path = os.path.join(pdf_root_folder, folder)

    if os.path.isdir(folder_path):  # Vérifie si c'est bien un dossier
        print(f"Traitement du dossier : {folder}")

        output_folder = os.path.join(output_root_folder, folder)
        os.makedirs(output_folder, exist_ok=True)

        for pdf_file in sorted(os.listdir(folder_path)):
            if pdf_file.endswith(".pdf"):
                file_path = os.path.join(folder_path, pdf_file)
                print(f"Traitement OCR : {file_path}")

                with open(file_path, "rb") as f:
                    response = requests.post(
                        OCR_URL,
                        files={"file": f},
                        data={
                            "apikey": OCR_API_KEY,
                            "language": "eng",
                            "isOverlayRequired": False,
                        },
                    )

                try:
                    result = response.json()
                except Exception as e:
                    print(f"Erreur de parsing JSON pour {pdf_file} : {e}")
                    result = None

                # Si le résultat est une chaîne, tenter de le recharger
                if result and isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except Exception as e:
                        print(
                            f"Erreur lors du chargement du JSON pour {pdf_file} : {e}"
                        )
                        result = None

                # Debug : Afficher le JSON obtenu si besoin
                # print(json.dumps(result, indent=4))

                # Vérifier si on a bien obtenu un résultat JSON valide
                if result:
                    if result.get("OCRExitCode") == 1:
                        extracted_text = result["ParsedResults"][0]["ParsedText"]

                        # Sauvegarder le texte extrait
                        text_file = os.path.join(
                            output_folder, pdf_file.replace(".pdf", ".txt")
                        )
                        with open(text_file, "w", encoding="utf-8") as f:
                            f.write(extracted_text)

                        print(f"Texte extrait et sauvegardé : {text_file}")
                    else:
                        print(
                            f"Erreur OCR pour {pdf_file} :",
                            result.get("ErrorMessage", "Erreur inconnue"),
                        )
                else:
                    print(
                        f"Réponse JSON invalide pour {pdf_file}. Réponse brute : {response.text}"
                    )
