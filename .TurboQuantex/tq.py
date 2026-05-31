import os
import sys
import argparse
import pickle
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from turboquantex import TurboQuantex

# Exclude directories by default
DEFAULT_EXCLUDES = {
    '.git', 'node_modules', 'vendor', 'storage', 'bootstrap', 
    'dist', 'build', '__pycache__', '.idea', '.vscode', '.venv',
    '.TurboQuantex'
}

# Exclude binary or non-code file extensions
IGNORED_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', 
    '.tar', '.gz', '.db', '.sqlite', '.exe', '.dll', '.so', '.bin'
}

import urllib.request
import json

_local_model_cache = None

def get_local_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Queries daemon server at http://127.0.0.1:59402 if running, or falls back to SentenceTransformer locally."""
    try:
        url = "http://127.0.0.1:59402/api/embed"
        req = urllib.request.Request(
            url, 
            data=json.dumps({"texts": texts}).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=2.0) as response:
            if response.status == 200:
                res_data = json.loads(response.read().decode('utf-8'))
                if res_data.get("status") == "success":
                    return [np.array(e, dtype=np.float32) for e in res_data["embeddings"]]
    except Exception:
        pass # Fail silently and fallback to local model load
        
    from sentence_transformers import SentenceTransformer
    global _local_model_cache
    if _local_model_cache is None:
        _local_model_cache = SentenceTransformer('all-MiniLM-L6-v2')
    embs = _local_model_cache.encode(texts)
    return [np.array(e, dtype=np.float32) for e in embs]

class CodebaseIndexer:
    def __init__(self, chunk_size: int = 1200, overlap: int = 200, extensions: List[str] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.extensions = extensions

    def chunk_file(self, file_path: str, rel_path: str) -> List[Dict[str, Any]]:
        """Splits a source file into logical chunks based on class/function structures (Python, PHP, etc.) with line metadata."""
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: Failed to read {rel_path}: {e}")
            return []

        if not lines:
            return []

        ext = os.path.splitext(file_path)[1].lower()
        is_code = ext in {'.py', '.php', '.js', '.ts', '.go', '.java', '.cpp', '.c', '.cs'}
        
        current_chunk_lines = []
        current_chunk_len = 0
        start_line = 1
        active_context = ""
        
        for line_idx, line in enumerate(lines):
            line_num = line_idx + 1
            line_len = len(line)
            
            # Detect logical boundaries
            is_boundary = False
            if is_code:
                trimmed = line.strip()
                if ext == '.py':
                    if trimmed.startswith(('def ', 'class ')):
                        is_boundary = True
                        active_context = trimmed.split('(')[0] if '(' in trimmed else trimmed
                elif ext in ('.php', '.js', '.ts', '.java', '.cs', '.go'):
                    if 'class ' in trimmed or 'function ' in trimmed or 'interface ' in trimmed:
                         is_boundary = True
                         active_context = trimmed.split('{')[0] if '{' in trimmed else trimmed
                         
            # Trigger boundary splitting if the accumulated chunk is long enough
            if is_boundary and current_chunk_len > 400:
                chunk_text = "".join(current_chunk_lines)
                context_header = f"File: {rel_path} (Lines {start_line}-{line_num - 1})\n"
                if active_context:
                    context_header += f"Context: {active_context}\n"
                context_header += "\n"
                
                chunks.append({
                    "file_path": rel_path,
                    "start_line": start_line,
                    "end_line": line_num - 1,
                    "text": chunk_text,
                    "embedding_text": context_header + chunk_text
                })
                current_chunk_lines = []
                current_chunk_len = 0
                start_line = line_num
                
            current_chunk_lines.append(line)
            current_chunk_len += line_len
            
            # Force cut if chunk exceeds maximum size
            if current_chunk_len >= self.chunk_size:
                chunk_text = "".join(current_chunk_lines)
                context_header = f"File: {rel_path} (Lines {start_line}-{line_num})\n"
                if active_context:
                    context_header += f"Context: {active_context}\n"
                context_header += "\n"
                
                chunks.append({
                    "file_path": rel_path,
                    "start_line": start_line,
                    "end_line": line_num,
                    "text": chunk_text,
                    "embedding_text": context_header + chunk_text
                })
                
                # Overlap implementation
                overlap_lines_count = max(1, int(len(current_chunk_lines) * (self.overlap / self.chunk_size)))
                current_chunk_lines = current_chunk_lines[-overlap_lines_count:]
                current_chunk_len = sum(len(l) for l in current_chunk_lines)
                start_line = line_num - len(current_chunk_lines) + 1

        if current_chunk_lines:
            chunk_text = "".join(current_chunk_lines)
            context_header = f"File: {rel_path} (Lines {start_line}-{len(lines)})\n"
            if active_context:
                context_header += f"Context: {active_context}\n"
            context_header += "\n"
            
            chunks.append({
                "file_path": rel_path,
                "start_line": start_line,
                "end_line": len(lines),
                "text": chunk_text,
                "embedding_text": context_header + chunk_text
            })
            
        return chunks

    def scan_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Recursively scans the directory and gathers text chunks."""
        all_chunks = []
        for root, dirs, files in os.walk(dir_path):
            # Exclude specified directories in-place to avoid descending into them
            dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDES]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in IGNORED_EXTENSIONS:
                    continue
                if self.extensions and ext not in self.extensions:
                    continue
                
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dir_path)
                
                file_chunks = self.chunk_file(full_path, rel_path)
                all_chunks.extend(file_chunks)
                
        return all_chunks

def run_indexing(args):
    print(f"[*] Scanning codebase: {args.dir}")
    
    # Resolve extensions filter
    exts = None
    if args.extensions:
        exts = [e.strip().lower() for e in args.extensions.split(",")]
        if not all(e.startswith(".") for e in exts):
            print("Error: Extensions must start with a dot (e.g. .py,.php)")
            sys.exit(1)
            
    indexer = CodebaseIndexer(chunk_size=args.chunk_size, overlap=args.overlap, extensions=exts)
    chunks = indexer.scan_directory(args.dir)
    n_chunks = len(chunks)
    print(f"[+] Found {n_chunks} code chunks across source files.")
    
    if n_chunks == 0:
        print("[-] No matching files found to index.")
        return

    dim = 384

    # Adaptive bit selection
    actual_bits = args.bits
    if args.bits == "auto" or args.bits is None:
        if n_chunks < 150:
            actual_bits = 4
        elif n_chunks < 450:
            actual_bits = 3
        else:
            actual_bits = 2
    else:
        try:
            actual_bits = int(args.bits)
        except ValueError:
            actual_bits = 2

    # Initialize TurboQuantex Compressor
    print(f"[*] Initializing TurboQuantex ({actual_bits}-bits, QJL={args.use_qjl}, QJL_dim={args.qjl_dim})...")
    engine = TurboQuantex(
        dim=dim,
        bits=actual_bits,
        use_qjl=args.use_qjl,
        qjl_dim=args.qjl_dim,
        seed=42
    )

    documents = []
    
    print("[*] Generating embeddings and compressing codebase...")
    t_start = time.time()
    
    # Batch query local embeddings
    chunk_texts = [chunk["embedding_text"] for chunk in chunks]
    embs = get_local_embeddings(chunk_texts)
    
    for idx, chunk in enumerate(chunks):
        if (idx + 1) % 50 == 0 or idx == n_chunks - 1:
            print(f"    Compressing chunk {idx + 1}/{n_chunks}...", end="\r")
            
        emb = embs[idx]
                
        # 2. Compress via TurboQuantex
        norm_x, indices, q_res, norm_res = engine.compress(emb)
        
        # Pack boolean array into uint8 bytes for binary disk storage efficiency
        q_res_packed = None
        if q_res is not None:
            q_res_packed = np.packbits(q_res)
            
        # Store compressed document representation (without original embedding to save disk space!)
        documents.append({
            "id": str(idx + 1),
            "file_path": chunk["file_path"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "text": chunk["text"],
            "norm_x": norm_x,
            "indices": indices.astype(np.uint8), # indices are between 0 and 15, so uint8 is plenty
            "q_res_packed": q_res_packed,
            "norm_res": norm_res
        })
        
    t_total = time.time() - t_start
    print(f"\n[+] Processing completed in {t_total:.2f} seconds.")

    # Cache file modification times
    file_meta = {}
    for doc in documents:
        rel_path = doc["file_path"]
        if rel_path not in file_meta:
            full_path = os.path.join(args.dir, rel_path)
            file_meta[rel_path] = os.path.getmtime(full_path) if os.path.exists(full_path) else 0.0

    # Save to disk using pickle
    index_data = {
        "config": {
            "embedding_mode": "local",
            "dim": dim,
            "bits": actual_bits,
            "use_qjl": args.use_qjl,
            "qjl_dim": args.qjl_dim,
            "seed": 42
        },
        "file_meta": file_meta,
        "documents": documents
    }
    
    with open(args.index, "wb") as f:
        pickle.dump(index_data, f)
        
    print(f"[+] Saved TurboQuantex codebase index to: {args.index}")
    show_stats_data(index_data, args.index)

def run_update(args):
    print(f"[*] Incrementally updating codebase index: {args.index}")
    from turboquantex_skill import update_codebase
    try:
        res = update_codebase(
            dir_path=args.dir,
            index_file=args.index
        )
        print(f"[+] Update status: {res['status']}")
        
        # Load updated index for stats printing
        with open(args.index, "rb") as f:
            index_data = pickle.load(f)
        show_stats_data(index_data, args.index)
    except Exception as e:
        print(f"[-] Error updating codebase: {e}")
        sys.exit(1)

def run_search(args):
    if not os.path.exists(args.index):
        print(f"Error: Index file '{args.index}' does not exist. Run 'index' command first.")
        sys.exit(1)
        
    # Load index data
    with open(args.index, "rb") as f:
        index_data = pickle.load(f)
        
    config = index_data["config"]
    documents = index_data["documents"]
    
    if not documents:
        print("[-] Index contains no documents.")
        return
        
    # Local mode query embedding
    embs = get_local_embeddings([args.query])
    query_emb = embs[0]

    # Initialize TurboQuantex engine for search similarity reconstruction
    engine = TurboQuantex(
        dim=config["dim"],
        bits=config["bits"],
        use_qjl=config["use_qjl"],
        qjl_dim=config["qjl_dim"],
        seed=config["seed"]
    )

    # Normalize query for cosine similarity estimation
    query_norm = np.linalg.norm(query_emb)
    query_norm_u = query_emb / (query_norm + 1e-8) if query_norm > 1e-8 else query_emb

    # Compute matches
    results = []
    for doc in documents:
        # Unpack boolean array from packed bits
        q_res = None
        if doc["q_res_packed"] is not None:
            # Unpack and slice to the original QJL dimensions
            q_res = np.unpackbits(doc["q_res_packed"])[:config["qjl_dim"]].astype(bool)
            
        sim = engine.estimate_inner_product(
            doc["norm_x"],
            doc["indices"].astype(np.int32),
            q_res,
            doc["norm_res"],
            query_norm_u
        )
        sim = float(np.clip(sim, -1.0, 1.0))
        results.append((doc, sim))
        
    # Sort results by similarity score
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:args.top_k]
    
    print(f"\n[+] Top {len(top_results)} matches for: '{args.query}'")
    print("=" * 80)
    for idx, (doc, score) in enumerate(top_results):
        print(f"\nRank #{idx + 1} | Similarity: {score:.4f} | {doc['file_path']} (Lines {doc['start_line']}-{doc['end_line']})")
        print("-" * 80)
        # Highlight code formatting with indentation
        indented_text = "\n".join("  " + l for l in doc["text"].split("\n")[:15])
        print(indented_text)
        if len(doc["text"].split("\n")) > 15:
            print("  ...")
        print("=" * 80)

def show_stats_data(index_data: Dict[str, Any], filepath: str):
    config = index_data["config"]
    documents = index_data["documents"]
    n_docs = len(documents)
    dim = config["dim"]
    
    disk_bytes = os.path.getsize(filepath)
    
    # Calculate original float32 storage cost
    original_bytes = n_docs * dim * 4
    
    # Calculate theoretical TurboQuantex compressed size
    bits_per_vector = 32 + 32 + (dim * config["bits"])
    if config["use_qjl"]:
        bits_per_vector += config["qjl_dim"]
    bytes_per_vector = int(np.ceil(bits_per_vector / 8))
    theoretical_compressed_bytes = n_docs * bytes_per_vector
    
    ratio = float(original_bytes) / float(theoretical_compressed_bytes) if theoretical_compressed_bytes > 0 else 0.0
    savings = (1.0 - float(theoretical_compressed_bytes) / float(original_bytes)) * 100.0 if original_bytes > 0 else 0.0
    
    print("\n" + "=" * 50)
    print("           TURBOQUANTEX INDEX METRICS           ")
    print("=" * 50)
    print(f"File Path:                {filepath}")
    print(f"Total Code Chunks:        {n_docs}")
    print(f"Vector Dimensions:        {dim}")
    print(f"Embedding Mode:           local")
    print(f"Quantization Bits:        {config['bits']} bits")
    print(f"QJL Correction:           {'Active' if config['use_qjl'] else 'Inactive'}")
    if config['use_qjl']:
        print(f"QJL Sketch Size:          {config['qjl_dim']} bits")
    print("-" * 50)
    print(f"Original Float32 RAM:     {original_bytes:,} Bytes ({original_bytes / 1024:.2f} KB)")
    print(f"TurboQuantex Packed RAM:  {theoretical_compressed_bytes:,} Bytes ({theoretical_compressed_bytes / 1024:.2f} KB)")
    print(f"Compressed Disk Size:     {disk_bytes:,} Bytes ({disk_bytes / 1024:.2f} KB)")
    print(f"Theoretical RAM Ratio:    {ratio:.2f}x compression")
    print(f"Theoretical RAM Savings:  {savings:.2f}% memory reduction")
    print("=" * 50)

def run_stats(args):
    if not os.path.exists(args.index):
        print(f"Error: Index file '{args.index}' does not exist. Run 'index' command first.")
        sys.exit(1)
        
    with open(args.index, "rb") as f:
        index_data = pickle.load(f)
        
    show_stats_data(index_data, args.index)

def main():
    parser = argparse.ArgumentParser(
        description="TurboQuantex CLI: Memory-Efficient Codebase Indexing & Semantic Search (Local Offline)"
    )
    subparsers = parser.add_subparsers(dest="command", help="CLI commands")
    
    # Subparser for index
    parser_index = subparsers.add_parser("index", help="Index a codebase directory")
    parser_index.add_argument("--dir", required=True, help="Path to directory containing source code files")
    parser_index.add_argument("--index", default="codebase_index.tq", help="Output file path for the compressed index (.tq)")
    parser_index.add_argument("--extensions", help="Comma-separated file extensions to include (e.g. .py,.php)")
    parser_index.add_argument("--bits", default="auto", help="Quantization bits (2, 3, 4, or 'auto' for dynamic selection)")
    parser_index.add_argument("--use-qjl", type=bool, default=True, help="Enable 1-bit QJL residual correction")
    parser_index.add_argument("--qjl-dim", type=int, choices=[64, 128, 256], default=128, help="QJL sketch dimension size")
    parser_index.add_argument("--chunk-size", type=int, default=1200, help="Maximum character length per code chunk")
    parser_index.add_argument("--overlap", type=int, default=200, help="Character overlap between consecutive chunks")
    
    # Subparser for update
    parser_update = subparsers.add_parser("update", help="Incrementally update an existing codebase index")
    parser_update.add_argument("--dir", required=True, help="Path to directory containing source code files")
    parser_update.add_argument("--index", default="codebase_index.tq", help="Index file path to update (.tq)")
 
    # Subparser for search
    parser_search = subparsers.add_parser("search", help="Perform semantic search on indexed codebase")
    parser_search.add_argument("--index", default="codebase_index.tq", help="Path to the index file (.tq)")
    parser_search.add_argument("--query", required=True, help="Query text for semantic search")
    parser_search.add_argument("--top-k", type=int, default=5, help="Number of matching snippets to return")
    
    # Subparser for stats
    parser_stats = subparsers.add_parser("stats", help="Show index details and compression metrics")
    parser_stats.add_argument("--index", default="codebase_index.tq", help="Path to the index file (.tq)")
    
    args = parser.parse_args()
    
    if args.command == "index":
        run_indexing(args)
    elif args.command == "update":
        run_update(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "stats":
        run_stats(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
