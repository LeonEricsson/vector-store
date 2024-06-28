from sentence_transformers import SentenceTransformer
from documents import document
from storage import VectorStorage

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

docs = document.split('\n')
print(f"Building vectorstore for {len(docs)} documents...")
print()

vs = VectorStorage(embedder=model)

vs.index(docs)

query = "What did emma do in this story?"
top_k_indices, scores = vs.search_top_k([query], k=3)

print(f"The most similar documents to the query '{query}' is:")
print()
print(f"{docs[top_k_indices[0][0]]}")
print()
print(f"w/ a similarity score of {scores[0][0]}")