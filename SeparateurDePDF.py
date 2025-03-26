import requests
import os
import base64

API_KEY = "API"

input_folder = "pdfNonSplit"

output_main_folder = "split_pages"
os.makedirs(output_main_folder, exist_ok=True)

API_URL = f"https://v2.convertapi.com/convert/pdf/to/split?Secret={API_KEY}"
# Pour chacun des dossiers, on ouvre et on feuillette les pdf, on les ouvre, puis on récupère chaque groupe de 3 pages de pdf en un seul vers le dossier split_pdf
for pdf_file in sorted(os.listdir(input_folder)):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(input_folder, pdf_file)
        print(f"🔍 Traitement du fichier : {pdf_file}")

        with open(pdf_path, "rb") as f:
            response = requests.post(
                API_URL,
                files={"file": f},
                data={"PageGroups": "3"},
            )

        if response.status_code == 200:
            result = response.json()

            if "Files" in result:
                pdf_output_folder = os.path.join(
                    output_main_folder, os.path.splitext(pdf_file)[0]
                )
                os.makedirs(pdf_output_folder, exist_ok=True)

                for i, file_info in enumerate(result["Files"]):
                    if "FileData" in file_info:
                        file_data = file_info["FileData"]
                        file_bytes = base64.b64decode(file_data)

                        file_path = os.path.join(
                            pdf_output_folder, f"group_{i + 1}.pdf"
                        )
                        with open(file_path, "wb") as f:
                            f.write(file_bytes)

                        print(f"Groupe {i + 1} de {pdf_file} sauvegardé : {file_path}")
                    else:
                        print(
                            f"Clé 'FileData' absente pour le groupe {i + 1} de {pdf_file}"
                        )

            else:
                print(
                    f"Erreur : La clé 'Files' est absente dans la réponse API pour {pdf_file} :",
                    result,
                )

        else:
            print(f"Erreur lors du split de {pdf_file} :", response.text)
