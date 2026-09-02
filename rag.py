import os

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


# Configuration 

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DOCS_DIR = "docs"
PDF_FILES = [
    "chapter1.pdf",
    "chapter2.pdf",
]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Load and chunk documents

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

all_chunks = []
all_metadata = []

for pdf_file in PDF_FILES:
    pdf_path = os.path.join(DOCS_DIR, pdf_file)

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    chunks = splitter.split_text(text)

    for chunk in chunks:
        all_chunks.append(chunk)
        all_metadata.append({
            "source": pdf_file
        })

print(f"Loaded {len(PDF_FILES)} PDFs")
print(f"Created {len(all_chunks)} chunks")


# Embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(all_chunks)


# Chroma vector store

chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection(
    name="data_structures"
)


ids = [f"chunk_{i}" for i in range(len(all_chunks))]

collection.upsert(
    ids=ids,
    documents=all_chunks,
    embeddings=embeddings.tolist(),
    metadatas=all_metadata,
)

print(f"Stored {len(all_chunks)} chunks in Chroma")


# RAG question answering

def ask(question: str) -> str:
    question_embedding = model.encode([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=3,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for document, metadata in zip(documents, metadatas):
        context_parts.append(
            f"Source: {metadata['source']}\n"
            f"{document}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a helpful Data Structures study assistant.

Answer the user's question using the provided course documentation.

The documentation is authoritative, but you may synthesize information
across retrieved passages and recognize related terminology when the
connection is supported by the provided context.

Do not invent facts that are not supported by the documentation.

If the retrieved context does not contain enough information to answer
the question reliably, say that you don't know based on the provided
course documentation.

Context:

""" + context

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response.choices[0].message.content


# Manual test
if __name__ == "__main__":
    answer = ask("What is a stack?")
    print("\n--- ANSWER ---")
    print(answer)

