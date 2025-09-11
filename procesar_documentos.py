import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

def procesar_documentos(api_key: str, carpeta="documentos", carpeta_index="faiss_index"):
    documentos = []

    # Leer todos los PDFs en la carpeta
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".pdf"):
            ruta = os.path.join(carpeta, archivo)
            loader = PyPDFLoader(ruta)
            documentos.extend(loader.load())

    if not documentos:
        raise ValueError("❌ No se encontraron documentos PDF en la carpeta.")

    # Dividir en fragmentos
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = splitter.split_documents(documentos)

    # Crear embeddings y FAISS
    embeddings = OpenAIEmbeddings(api_key=api_key)
    db = FAISS.from_documents(fragmentos, embeddings)

    # Guardar el índice
    db.save_local(carpeta_index)

    return db
