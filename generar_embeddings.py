from procesar_documentos import procesar_documentos
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Extraer texto de documentos
texto_total = procesar_documentos("documentos")

# Dividir en fragmentos
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
fragmentos = splitter.split_text(texto_total)

# Crear embeddings
embeddings = OpenAIEmbeddings()

# Crear base vectorial
vectorstore = FAISS.from_texts(fragmentos, embeddings)

# Guardar la base vectorial
vectorstore.save_local("base_vectorial")
print("✅ Base vectorial generada y guardada.")
