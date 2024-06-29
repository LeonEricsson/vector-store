import timeit
from sentence_transformers import SentenceTransformer
from data.paulgraham import document
from storage import VectorStorage


model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=512)

docs = document.split('\n')
print(f"Building vectorstore for {len(docs)} documents...")
print()

query_prefix = 'Represent this sentence for searching relevant passages: '

vs = VectorStorage(embedder=model, query_prefix=query_prefix)

vs.index(docs)

query = "What was the most popular forum before Reddit?"

run_search = lambda query: vs.search_top_k([query], k=3)

runs = 100
time = timeit.timeit(lambda: run_search(query), number=runs)

top_k_indices, scores = run_search(query)

print(f"Execution time {(time/runs) * 1000:.4f} ms")
print()
print(f"The most similar documents to the query '{query}' is:")
print()
print(f"{docs[top_k_indices[0][0]]}")
print()
print(f"w/ a similarity score of {scores[0][0]}")