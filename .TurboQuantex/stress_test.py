import time
import numpy as np
import os
import sys

# Append path to import turboquantex core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from turboquantex import TurboQuantex

def run_stress_test(num_vectors=10000, dim=384, bits=3, use_qjl=True, qjl_dim=128):
    print("==================================================")
    print(f"       TURBOQUANTEX ENGINE STRESS TEST            ")
    print("==================================================")
    
    # 1. Generate synthetic vectors (L2 normalized to represent embeddings)
    t0 = time.time()
    raw_vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    raw_vectors = raw_vectors / (norms + 1e-9)
    
    # Generate random normalized query vector
    query_vector = np.random.randn(dim).astype(np.float32)
    query_vector /= (np.linalg.norm(query_vector) + 1e-9)
    print(f"[+] Generated {num_vectors:,} synthetic vectors ({dim} dims) in {time.time() - t0:.4f}s")
    
    # Initialize Engine
    engine = TurboQuantex(dim=dim, bits=bits, use_qjl=use_qjl, qjl_dim=qjl_dim, seed=42)
    
    # 2. Measure Compression Speed
    t_comp_start = time.time()
    compressed_database = []
    for vec in raw_vectors:
        norm_x, indices, q_res, norm_res = engine.compress(vec)
        compressed_database.append((norm_x, indices, q_res, norm_res))
    t_comp = time.time() - t_comp_start
    print(f"[+] Compressed {num_vectors:,} vectors in {t_comp:.4f}s ({num_vectors/t_comp:.1f} vectors/sec)")
    
    # 3. True Float32 Similarity (Exact Math)
    t_exact_start = time.time()
    true_similarities = np.dot(raw_vectors, query_vector)
    t_exact = time.time() - t_exact_start
    
    # 4. TurboQuantex Estimated Similarity (Estimated Math)
    t_est_start = time.time()
    estimated_similarities = []
    for norm_x, indices, q_res, norm_res in compressed_database:
        sim = engine.estimate_inner_product(norm_x, indices, q_res, norm_res, query_vector)
        estimated_similarities.append(sim)
    estimated_similarities = np.array(estimated_similarities, dtype=np.float32)
    t_est = time.time() - t_est_start
    print(f"[+] Searched {num_vectors:,} compressed vectors in {t_est:.4f}s ({num_vectors/t_est:.1f} vectors/sec)")
    
    # 5. Statistical Quality & Distortion Metrics
    mse = np.mean((true_similarities - estimated_similarities) ** 2)
    correlation = np.corrcoef(true_similarities, estimated_similarities)[0, 1]
    
    # 6. Memory Footprint Savings
    original_bytes = num_vectors * dim * 4
    
    # Dynamic pack size calculation per vector
    bits_per_vector = 32 + 32 + (dim * bits)
    if use_qjl:
        bits_per_vector += qjl_dim
    bytes_per_vector = int(np.ceil(bits_per_vector / 8))
    turbo_bytes = num_vectors * bytes_per_vector
    ratio = original_bytes / turbo_bytes
    savings = (1 - (turbo_bytes / original_bytes)) * 100
    
    print("\n---------------- RESULTS ----------------")
    print(f"Original RAM size:   {original_bytes:,} Bytes ({original_bytes/1024/1024:.2f} MB)")
    print(f"Compressed RAM size: {turbo_bytes:,} Bytes ({turbo_bytes/1024/1024:.2f} MB)")
    print(f"Compression Ratio:   {ratio:.2f}x reduction")
    print(f"RAM Savings:         {savings:.2f}% memory saved")
    print(f"Mean Squared Error:  {mse:.6f} (QJL residual distortion)")
    print(f"Pearson Correlation: {correlation:.6f} (Reconstruction accuracy)")
    print("==================================================")

if __name__ == "__main__":
    # Test with 10,000 vectors
    run_stress_test(num_vectors=10000, bits=3)
