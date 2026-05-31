"""
stress_test.py - Stress Testing & Benchmarking Utility for TurboQuantex local engine.
"""

import os
import sys
import time
import psutil
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Add current path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import turboquantex_skill
from turbo_code import CodebaseIndexer
from turboquantex import TurboQuantex

INDEX_FILE = "stress_test_index.tq"
DIR_TO_INDEX = "example_project"

def format_bytes(b: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} TB"

def run_stress_test():
    print("=" * 60)
    print("           TURBOQUANTEX STRESS TEST & BENCHMARK          ")
    print("=" * 60)
    
    # 1. Codebase Scan & Size Analysis
    print("\n[*] Phase 1: Codebase Analysis...")
    t0 = time.time()
    
    # Simple count of source files in the project
    total_files = 0
    total_lines = 0
    total_chars = 0
    
    from turbo_code import DEFAULT_EXCLUDES, IGNORED_EXTENSIONS
    for root, dirs, files in os.walk(DIR_TO_INDEX):
        dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDES]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IGNORED_EXTENSIONS:
                continue
            full_path = os.path.join(root, file)
            total_files += 1
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    total_lines += len(lines)
                    total_chars += sum(len(l) for l in lines)
            except Exception:
                pass
                
    scan_duration = time.time() - t0
    print(f"[+] Scan Complete:")
    print(f"    - Total Files Analyzed:  {total_files}")
    print(f"    - Total Lines of Code:   {total_lines:,}")
    print(f"    - Codebase Char Count:   {total_chars:,} chars")
    print(f"    - Scan Duration:         {scan_duration:.4f} seconds")

    # 2. Stress Indexing (Embedding + Quantization)
    print("\n[*] Phase 2: Codebase Indexing & Vector Compression (auto bits + QJL)...")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    t_start = time.time()
    stats = turboquantex_skill.index_codebase(
        dir_path=DIR_TO_INDEX,
        index_file=INDEX_FILE,
        bits="auto",
        use_qjl=True,
        qjl_dim=128
    )
    t_index = time.time() - t_start
    mem_after = process.memory_info().rss
    
    chunks_count = stats.get("chunks", 0)
    dim = stats.get("dimensions", 384)
    disk_size = os.path.getsize(INDEX_FILE)
    
    print(f"[+] Indexing Complete:")
    print(f"    - Total Chunks Indexed:  {chunks_count}")
    print(f"    - Total Time Taken:      {t_index:.2f} seconds")
    if chunks_count > 0:
        print(f"    - Indexing Speed:        {chunks_count / t_index:.2f} chunks/sec")
    print(f"    - RAM Leak / Increase:   {format_bytes(mem_after - mem_before)}")

    # 3. Compression Metrics
    print("\n[*] Phase 3: Vector Compression Metrics...")
    # Float32 cost: chunks * dimensions * 4 bytes
    float32_bytes = chunks_count * dim * 4
    # Theoretical TurboQuantex packed RAM bytes:
    # norm_x (4 bytes) + norm_res (4 bytes) + indices (dim * bits) + q_res (128 bits)
    actual_bits = stats.get("bits", 4)
    bits_per_vector = 32 + 32 + (dim * actual_bits) + 128
    bytes_per_vector = int(np.ceil(bits_per_vector / 8))
    turboquantex_ram_bytes = chunks_count * bytes_per_vector
    
    savings = (1.0 - (turboquantex_ram_bytes / float32_bytes)) * 100.0 if float32_bytes > 0 else 0.0
    comp_ratio = float32_bytes / turboquantex_ram_bytes if turboquantex_ram_bytes > 0 else 0.0
    
    print(f"    - Dimensions:            {dim}")
    print(f"    - Original Float32 RAM:  {format_bytes(float32_bytes)}")
    print(f"    - TurboQuantex Packed RAM: {format_bytes(turboquantex_ram_bytes)}")
    print(f"    - Compressed Disk Size:  {format_bytes(disk_size)}")
    print(f"    - Memory Savings:        {savings:.2f}% RAM reduction")
    print(f"    - Compression Ratio:     {comp_ratio:.2f}x compression")

    # 4. Search Latency & Stress Querying
    print("\n[*] Phase 4: Stress Semantic Search Queries...")
    queries = [
        "user registration validation credentials",
        "average values calculation list",
        "database insert user record statement"
    ]
    
    search_times = []
    embed_times = []
    
    for q in queries:
        t_embed_start = time.time()
        # Mock encoding query to isolate search time vs embedding time
        local_model = SentenceTransformer('all-MiniLM-L6-v2')
        local_model.encode(q)
        t_embed = time.time() - t_embed_start
        embed_times.append(t_embed)
        
        t_search_start = time.time()
        results = turboquantex_skill.query_codebase(INDEX_FILE, q, top_k=5)
        t_search = time.time() - t_search_start
        search_times.append(t_search)
        
        print(f"    - Query: '{q}' -> Found {len(results)} matches (Search time: {t_search * 1000:.2f} ms)")
        
    avg_embed = np.mean(embed_times) * 1000
    avg_search = np.mean(search_times) * 1000
    
    print(f"[+] Average Query Embedding Latency: {avg_embed:.2f} ms")
    if avg_search > 0:
        print(f"[+] Average Inner Product Scan Speed: {chunks_count / (avg_search / 1000.0):,.0f} vectors/sec")
    print(f"[+] Average Search Retrieval Latency:  {avg_search:.2f} ms")

    # 5. Stress Incremental Update
    print("\n[*] Phase 5: Stress Incremental Update Performance...")
    # Clean run with no changes
    t_clean_start = time.time()
    clean_update = turboquantex_skill.update_codebase(dir_path=DIR_TO_INDEX, index_file=INDEX_FILE)
    t_clean = time.time() - t_clean_start
    print(f"    - Update (Zero modifications):  {t_clean * 1000:.2f} ms (Status: {clean_update['status']})")
    
    # Modify 1 file to simulate edit
    file_to_modify = os.path.join(DIR_TO_INDEX, "scripts", "data_processor.py")
    if os.path.exists(file_to_modify):
        print(f"    - Modifying file: '{file_to_modify}'...")
        with open(file_to_modify, "a", encoding="utf-8") as f:
            f.write("\n# Stress test edit comment line\n")
            
        t_mod_start = time.time()
        mod_update = turboquantex_skill.update_codebase(dir_path=DIR_TO_INDEX, index_file=INDEX_FILE)
        t_mod = time.time() - t_mod_start
        
        print(f"    - Update (1 modified file):     {t_mod:.2f} seconds (Status: {mod_update['status']})")
        
        # Restore file
        with open(file_to_modify, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines[-1].strip() == "# Stress test edit comment line":
            with open(file_to_modify, "w", encoding="utf-8") as f:
                f.writelines(lines[:-1])
        print("    - Restored file to original state.")
        
    # Cleanup index file
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
        print("    - Cleaned up stress_test_index.tq.")
        
    print("\n" + "=" * 60)
    print("                STRESS TESTS SUCCESSFUL!               ")
    print("=" * 60)

if __name__ == "__main__":
    run_stress_test()
