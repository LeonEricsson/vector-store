import numpy as np
import timeit
import csv
import os

from sentence_transformers import SentenceTransformer
from storage import VectorStorage

E = 512  # Embedding dimension for the model
model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1', truncate_dim=E)

def run_benchmark(N_s, k=20):
    random_vectors = np.random.rand(N_s, E).astype(np.float32)
    vs = VectorStorage(embedder=model)
    vs._index = random_vectors

    queries = ["What was the most popular forum before Reddit?"]
    
    run_search = lambda: vs.search_top_k(queries, k=k)
    
    runs = 20
    time = timeit.timeit(run_search, number=runs)
    
    return (time / runs) * 1000  # Convert to milliseconds

# Define ranges for N_s
N_s_range = [1e2, 1e3, 1e4, 1e5, 2.5e5, 5e5, 1e6]

# Run benchmarks and store results
results = []
csv_filename = 'vs_benchmark.csv'

# Load existing results if the file exists
existing_results = set()
if os.path.exists(csv_filename):
    with open(csv_filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            if len(row) >= 2:
                existing_results.add(int(row[0]))

mode = 'a' if existing_results else 'w'
with open(csv_filename, mode, newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if mode == 'w':
        csvwriter.writerow(['N_s', 'Search_Time_ms'])

    for N_s in N_s_range:
        N_s = int(N_s)
        if N_s in existing_results:
            print(f"Skipping existing result: N_s={N_s}")
            continue

        print(f"Running benchmark: N_s={N_s}")
        
        search_time = run_benchmark(N_s)
        results.append([N_s, search_time])
        csvwriter.writerow([N_s, search_time])
        print(f"Completed benchmark: N_s={N_s}, Time={search_time:.2f}ms")
        csvfile.flush()  # Ensure data is written immediately
        

print(f"Results saved to {csv_filename}")