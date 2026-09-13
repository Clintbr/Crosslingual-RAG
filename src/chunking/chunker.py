"""
Dieses Modul implementiert die Dokumenten-Ingestion-Pipeline für die Chunk-Datenbank.

Ablauf:

1. MongoDB-Verbindung:
   Es wird eine Verbindung zur MongoDB aufgebaut (MONGO_URI, DB_NAME) und zwei
   Collections referenziert:
     - "chunks": speichert die einzelnen Text-Chunks
     - "processed_files": speichert Metadaten zu bereits verarbeiteten Dateien
       (dient als Deduplizierungs-Register)

2. Text-Splitter-Konfiguration:
   Ein RecursiveCharacterTextSplitter wird mit chunk_size=800 und chunk_overlap=0
   initialisiert. Er versucht, Text bevorzugt an Absätzen ("\n\n") und Zeilenumbrüchen
   ("\n") zu trennen, um möglichst sinnvolle (nicht mitten im Satz abgeschnittene)
   Chunks zu erzeugen.

3. get_file_hash(filepath):
   Berechnet einen MD5-Hash über den kompletten Dateiinhalt. Dieser Hash dient als
   eindeutiger Fingerabdruck der Datei, um bereits verarbeitete Dokumente zu erkennen
   (unabhängig vom Dateinamen).

4. chunk_document(filepath):
   Hauptfunktion der Pipeline, verarbeitet eine einzelne Datei:

   a) Vorbereitung:
      - Extrahiert den Dateinamen und berechnet den Datei-Hash.
      - Prüft in "processed_files", ob dieser Hash bereits existiert. Falls ja,
        wird die Verarbeitung übersprungen (Deduplizierung) und die Funktion bricht
        mit None ab.

   b) Textextraktion (Schritt 1):
      - Nutzt `unstructured.partition.auto.partition`, um den Dateiinhalt in einzelne
        strukturierte Elemente zu zerlegen.
      - Für PDFs wird die Strategie "hi_res" verwendet (genauere, aber langsamere
        Analyse inkl. Layout-/OCR-Erkennung), für alle anderen Dateitypen "auto"
        (automatische Strategiewahl durch unstructured).

   c) Seitenweise Gruppierung (Schritt 2):
      - Die extrahierten Elemente enthalten Metadaten, u. a. die Seitenzahl
        (page_number). Der Text aller Elemente wird pro Seite zusammengefügt,
        sodass am Ende ein Dictionary {seitenzahl: gesamter_seitentext} entsteht.
      - Damit wird sichergestellt, dass Chunking später nicht seitenübergreifend
        erfolgt, sondern die Seitenzuordnung für jeden Chunk erhalten bleibt.

   d) Chunking pro Seite (Schritt 3):
      - Für jede Seite wird geprüft, ob der Text (nach Trimmen) mindestens 10 Zeichen
        lang ist; sehr kurze/leere Seiten werden übersprungen.
      - Der Seitentext wird mit dem RecursiveCharacterTextSplitter in kleinere,
        semantisch sinnvolle Chunks (max. 800 Zeichen) zerlegt.
      - Für jeden erzeugten Chunk wird ein MongoDB-Dokument mit folgenden Feldern
        gebaut: Dateiname, Datei-Hash, Chunk-Inhalt, Sprache (aus den ersten beiden
        Zeichen des Dateinamens abgeleitet, z. B. "en_..." -> "en"), Seitenzahl und
        fortlaufender Chunk-Index.
      - Das Dokument wird sofort in die "chunks"-Collection eingefügt; die von
        MongoDB vergebene _id wird anschließend dem lokalen Dict hinzugefügt und das
        Dokument der Rückgabeliste `chunks_mongo` angehängt.
      - Ein globaler Zähler `chunk_count` wird über alle Seiten hinweg hochgezählt,
        sodass jeder Chunk innerhalb der Datei einen eindeutigen, fortlaufenden Index
        erhält.

   e) Abschluss (Schritt 4):
      - Nach Verarbeitung aller Seiten wird ein Eintrag in "processed_files" angelegt
        (Dateiname, Hash, Sprache, Verarbeitungszeitpunkt). Dieser Eintrag verhindert
        bei einem erneuten Lauf, dass dieselbe Datei doppelt verarbeitet wird.
      - Es wird eine Erfolgsmeldung mit der Gesamtanzahl gespeicherter Chunks
        ausgegeben.
      - Die Funktion gibt die Liste aller erzeugten Chunk-Dokumente (inkl. Mongo-IDs)
        zurück.

Kurz zusammengefasst: Die Pipeline liest eine Datei ein, extrahiert und strukturiert
ihren Text seitenweise, teilt jede Seite in überlappungsfreie Chunks von max. 800
Zeichen, speichert jeden Chunk als eigenes Dokument in MongoDB und vermerkt die Datei
danach als "bereits verarbeitet", um Mehrfachverarbeitung zu vermeiden.
"""

import os
import hashlib
from pymongo import MongoClient
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pytesseract
from datetime import datetime
from src.config import (
    MONGO_URI, DB_NAME, TESSERACT_PATH, TESSERACT_EXEC_PATH
)

tesseract_exec_path = TESSERACT_EXEC_PATH

if os.path.exists(tesseract_exec_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_exec_path
    os.environ["PATH"] += os.pathsep + TESSERACT_PATH
else:
    print(f"WARNUNG: Tesseract wurde unter {tesseract_exec_path} nicht gefunden!")

m_client = MongoClient(MONGO_URI)
db = m_client[DB_NAME]
chunks_col = db["chunks"]
files_col = db["processed_files"]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=0,
    separators=["\n\n", "\n"]
)

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def chunk_document(filepath):
    filename = os.path.basename(filepath)
    file_hash = get_file_hash(filepath)
    chunk_count = 0
    chunks_mongo = []

    print(f"Start Ingestion for: {filename}...🟩")
    if files_col.find_one({"file_hash": file_hash}):
        print(f"Skipping process for {filename}: already ingested.🟠")
        print("Skipping chunking...🟠")
        return
    print(f"Extract text from: {filename}...🟩")

    # 1. text extraction with unstructured
    elements = partition(
        filename=filepath,
        strategy="hi_res" if filepath.endswith(".pdf") else "auto"
    )

    # 2. we group the text by pagenumber to allow a page based chunking
    print(f"Start chunking for: {filename}...🟩")
    pages = {}
    for el in elements:
        p = el.metadata.to_dict().get("page_number", 1)
        pages[p] = pages.get(p, "") + "\n" + str(el)

    # 3. Intelligent page-level chunking
    for page_num, page_text in pages.items():
        if len(page_text.strip()) < 10: continue

        # the splitter pays attention to paragraphs and sentences.
        chunks = text_splitter.split_text(page_text)

        for i, chunk_content in enumerate(chunks):
            # 3. save the chunks in MongoDB
            mongo_doc = {
                "filename": filename,
                "file_hash": file_hash,
                "content": chunk_content,
                "lang": filename[:2],
                "page": page_num,
                "chunk_index": chunk_count
            }
            m_id = str(chunks_col.insert_one(mongo_doc).inserted_id)
            mongo_doc.update({"_id": m_id})
            chunks_mongo.append(mongo_doc)
            chunk_count += 1

    # 4. save the file in MongoDB
    files_col.insert_one({
        "filename": filename,
        "file_hash": file_hash,
        "lang": filename[:2],
        "processed_at": datetime.now()
    })

    print(f"chunks saved to mongodb: {filename} -> {chunk_count} 🟩")

    return chunks_mongo