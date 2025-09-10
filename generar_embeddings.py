from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from procesar_documentos import procesar_documentos
import os

# Configurar clave de OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Procesar documentos
texto = procesar_documentos("documentos")

# Dividir en fragmentos
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
fragmentos = splitter.split_text(texto)

# Crear embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(fragmentos, embeddings)

# Guardar la base vectorial
vectorstore.save_local("base_vectorial")
print("✅ Base FAISS generada correctamente.")
