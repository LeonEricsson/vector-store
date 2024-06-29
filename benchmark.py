import numpy as np
import matplotlib.pyplot as plt
import timeit
import csv
import traceback
import os

from sentence_transformers import SentenceTransformer
from storage import VectorStorage

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
E = 384  # Embedding dimension for the model

def run_benchmark(N_s, N_q, k=20):
    random_vectors = np.random.rand(N_s, E).astype(np.float32)
    vs = VectorStorage(embedder=model)
    vs.index = random_vectors

    queries = ["What was the most popular forum before Reddit?"] * N_q
    
    #run_search = lambda: vs.search_top_k(queries, k=k)
    run_search = lambda: vs.search_faiss(queries, k=k)
    
    runs = 100
    time = timeit.timeit(run_search, number=runs)
    
    return (time / runs) * 1000  # Convert to milliseconds

# Define ranges for N_s and N_q
N_s_range = [100, 1000, 5000, 10000, 50000, 100000, 250000, 500000, 1000000]
N_q_range = [1]

# Create meshgrid
N_s_mesh, N_q_mesh = np.meshgrid(N_s_range, N_q_range)

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
                existing_results.add((int(row[0]), int(row[1])))

mode = 'a' if existing_results else 'w'
with open(csv_filename, mode, newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if mode == 'w':
        csvwriter.writerow(['N_s', 'N_q', 'Search_Time_ms'])

    for N_s in N_s_range:
        for N_q in N_q_range:
            if (N_s, N_q) in existing_results:
                print(f"Skipping existing result: N_s={N_s}, N_q={N_q}")
                continue

            print(f"Running benchmark: N_s={N_s}, N_q={N_q}")
            
            search_time = run_benchmark(N_s, N_q)
            results.append([N_s, N_q, search_time])
            csvwriter.writerow([N_s, N_q, search_time])
            print(f"Completed benchmark: N_s={N_s}, N_q={N_q}, Time={search_time:.2f}ms")
            csvfile.flush()  # Ensure data is written immediately
            

print(f"Results saved to {csv_filename}")