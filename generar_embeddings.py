from procesar_documentos import procesar_documentos
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Extraer texto de documentos
texto_total = procesar_documentos("documentos")

# Dividir en fragmentos
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
fragmentos = splitter.split_text(texto_total)

# Crear embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(fragmentos, embeddings)

# Guardar la base vectorial
vectorstore.save_local("base_vectorial")
print("✅ Base vectorial generada y guardada.")
