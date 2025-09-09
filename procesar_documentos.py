import os
import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup

def extraer_texto_pdf(ruta):
    doc = fitz.open(ruta)
    return "\n".join([pagina.get_text() for pagina in doc])

def extraer_texto_docx(ruta):
    doc = Document(ruta)
    return "\n".join([p.text for p in doc.paragraphs])

def extraer_texto_html(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text()

def procesar_documentos(carpeta="documentos"):
    textos = []
    for archivo in os.listdir(carpeta):
        ruta = os.path.join(carpeta, archivo)
        ext = os.path.splitext(archivo)[1].lower()
        try:
            if ext == ".pdf":
                textos.append(extraer_texto_pdf(ruta))
            elif ext == ".docx":
                textos.append(extraer_texto_docx(ruta))
            elif ext == ".html":
                textos.append(extraer_texto_html(ruta))
        except Exception as e:
            print(f"Error en {archivo}: {e}")
    return "\n".join(textos)
