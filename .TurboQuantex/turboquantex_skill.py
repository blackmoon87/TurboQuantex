"""
TurboQuantex Codebase Search Skill API - Local Offline Version

This module exposes reusable functions for indexing and searching codebases 
using TurboQuantex-compressed vector embeddings. It runs 100% locally and offline, 
without requiring any API keys.
"""

import os
import sys
import pickle
import numpy as np
from typing import List, Dict, Any

# Ensure the local path is included for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tq import CodebaseIndexer, INDEX_VERSION, INDEX_MODEL_ID
from turboquantex import TurboQuantex

def index_codebase(
    dir_path: str, 
    index_file: str = "codebase_index.tq", 
    bits: Any = "auto", 
    use_qjl: bool = True, 
    qjl_dim: int = 128
) -> Dict[str, Any]:
    """
    Scans, chunks, embeds, and compresses a codebase directory into a .tq file.
    Runs entirely locally using SentenceTransformers ('all-MiniLM-L6-v2').
    """
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory path '{dir_path}' does not exist.")
        
    indexer = CodebaseIndexer(chunk_size=1200, overlap=200)
    chunks = indexer.scan_directory(dir_path)
    n_chunks = len(chunks)
    
    if n_chunks == 0:
        return {"chunks": 0, "status": "No files found to index."}
        
    dim = 384
    from tq import get_local_embeddings
    
    # Adaptive bit selection
    actual_bits = bits
    if bits == "auto" or bits is None:
        if n_chunks < 150:
            actual_bits = 4
        elif n_chunks < 450:
            actual_bits = 3
        else:
            actual_bits = 2
    else:
        actual_bits = int(bits)
        
    # Initialize TurboQuantex
    engine = TurboQuantex(dim=dim, bits=actual_bits, use_qjl=use_qjl, qjl_dim=qjl_dim, seed=42)
    documents = []
    file_meta = {}
    
    # Batch query embeddings
    chunk_texts = [chunk["embedding_text"] for chunk in chunks]
    embs = get_local_embeddings(chunk_texts)
    
    for idx, chunk in enumerate(chunks):
        # Cache file modification time
        rel_path = chunk["file_path"]
        if rel_path not in file_meta:
            full_path = os.path.join(dir_path, rel_path)
            file_meta[rel_path] = os.path.getmtime(full_path) if os.path.exists(full_path) else 0.0
            
        emb = embs[idx]
        norm_x, indices, q_res, norm_res = engine.compress(emb)
        
        q_res_packed = None
        if q_res is not None:
            q_res_packed = np.packbits(q_res)
            
        documents.append({
            "id": str(idx + 1),
            "file_path": rel_path,
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "text": chunk["text"],
            "language": chunk.get("language", "unknown"),
            "scope": chunk.get("scope", ""),
            "norm_x": norm_x,
            "indices": indices.astype(np.uint8),
            "q_res_packed": q_res_packed,
            "norm_res": norm_res
        })
        
    index_data = {
        "version": INDEX_VERSION,
        "model_id": INDEX_MODEL_ID,
        "config": {
            "embedding_mode": "local",
            "dim": dim,
            "bits": actual_bits,
            "use_qjl": use_qjl,
            "qjl_dim": qjl_dim,
            "seed": 42
        },
        "file_meta": file_meta,
        "documents": documents
    }
    
    with open(index_file, "wb") as f:
        pickle.dump(index_data, f)
        
    return {
        "chunks": len(documents),
        "dimensions": dim,
        "mode": "local",
        "bits": actual_bits,
        "disk_size_kb": round(os.path.getsize(index_file) / 1024.0, 2),
        "status": "Success"
    }

def update_codebase(
    dir_path: str,
    index_file: str = "codebase_index.tq"
) -> Dict[str, Any]:
    """
    Incrementally updates the .tq index file locally. Re-indexes only modified or new files,
    and removes chunks corresponding to deleted files.
    """
    if not os.path.exists(index_file):
        return index_codebase(dir_path, index_file)
        
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory path '{dir_path}' does not exist.")
        
    # 1. Load existing index
    with open(index_file, "rb") as f:
        index_data = pickle.load(f)
        
    config = index_data["config"]
    file_meta = index_data.get("file_meta", {})
    documents = index_data["documents"]
    
    # Initialize indexer and scan active files
    indexer = CodebaseIndexer(chunk_size=1200, overlap=200)
    
    active_files = {}
    for root, dirs, files in os.walk(dir_path):
        from tq import DEFAULT_EXCLUDES, IGNORED_EXTENSIONS
        dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDES]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IGNORED_EXTENSIONS:
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, dir_path)
            active_files[rel_path] = os.path.getmtime(full_path)
 
    # 2. Find changes
    deleted_files = set(file_meta.keys()) - set(active_files.keys())
    modified_files = set()
    new_files = set(active_files.keys()) - set(file_meta.keys())
    
    for rel_path, mtime in active_files.items():
        if rel_path in file_meta:
            if abs(mtime - file_meta[rel_path]) > 1e-4:
                modified_files.add(rel_path)
                
    files_to_remove = deleted_files | modified_files
    files_to_index = new_files | modified_files
    
    if not files_to_remove and not files_to_index:
        return {
            "chunks": len(documents),
            "status": "Already up to date. No files modified.",
            "disk_size_kb": round(os.path.getsize(index_file) / 1024.0, 2)
        }
        
    # 3. Remove chunks of deleted/modified files
    documents = [doc for doc in documents if doc["file_path"] not in files_to_remove]
    for rel_path in files_to_remove:
        file_meta.pop(rel_path, None)
        
    # 4. Re-index modified/new files locally
    if files_to_index:
        dim = 384
        from tq import get_local_embeddings
            
        # Initialize TurboQuantex engine
        engine = TurboQuantex(
            dim=dim, 
            bits=config["bits"], 
            use_qjl=config["use_qjl"], 
            qjl_dim=config["qjl_dim"], 
            seed=config["seed"]
        )
        
        # Index new/modified chunks
        new_chunks = []
        for rel_path in files_to_index:
            full_path = os.path.join(dir_path, rel_path)
            file_chunks = indexer.chunk_file(full_path, rel_path)
            new_chunks.extend(file_chunks)
            
        if new_chunks:
            chunk_texts = [c["embedding_text"] for c in new_chunks]
            embs = get_local_embeddings(chunk_texts)
            
            for idx, chunk in enumerate(new_chunks):
                rel_path = chunk["file_path"]
                file_meta[rel_path] = active_files[rel_path]
                
                emb = embs[idx]
                norm_x, indices, q_res, norm_res = engine.compress(emb)
                
                q_res_packed = None
                if q_res is not None:
                    q_res_packed = np.packbits(q_res)
                    
                documents.append({
                    "id": str(len(documents) + 1),
                    "file_path": rel_path,
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "text": chunk["text"],
                    "norm_x": norm_x,
                    "indices": indices.astype(np.uint8),
                    "q_res_packed": q_res_packed,
                    "norm_res": norm_res
                })
            
    # 5. Save updated index
    index_data["documents"] = documents
    index_data["file_meta"] = file_meta
    
    with open(index_file, "wb") as f:
        pickle.dump(index_data, f)
        
    return {
        "chunks": len(documents),
        "added_files": len(files_to_index),
        "removed_files": len(files_to_remove),
        "disk_size_kb": round(os.path.getsize(index_file) / 1024.0, 2),
        "status": f"Successfully updated. Indexed {len(files_to_index)} files, removed {len(files_to_remove)} files."
    }

def query_codebase(
    index_file: str, 
    query: str, 
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Queries a TurboQuantex-compressed codebase index file semantically.
    Runs entirely locally using SentenceTransformers.
    """
    if not os.path.exists(index_file):
        raise ValueError(f"Index file '{index_file}' does not exist.")
        
    with open(index_file, "rb") as f:
        index_data = pickle.load(f)
    
    # Version check
    version = index_data.get("version", 1)
    if version < INDEX_VERSION:
        raise ValueError(f"Index version {version} is outdated (current: {INDEX_VERSION}). Please re-index.")
        
    config = index_data["config"]
    documents = index_data["documents"]
    
    if not documents:
        return []
        
    from tq import get_local_embeddings
    embs = get_local_embeddings([query])
    query_emb = embs[0]
        
    # Initialize engine
    engine = TurboQuantex(
        dim=config["dim"],
        bits=config["bits"],
        use_qjl=config["use_qjl"],
        qjl_dim=config["qjl_dim"],
        seed=config["seed"]
    )
    
    query_norm = np.linalg.norm(query_emb)
    query_norm_u = query_emb / (query_norm + 1e-8) if query_norm > 1e-8 else query_emb
    
    results = []
    for doc in documents:
        q_res = None
        if doc["q_res_packed"] is not None:
            q_res = np.unpackbits(doc["q_res_packed"])[:config["qjl_dim"]].astype(bool)
            
        sim = engine.estimate_inner_product(
            doc["norm_x"],
            doc["indices"].astype(np.int32),
            q_res,
            doc["norm_res"],
            query_norm_u
        )
        sim = float(np.clip(sim, -1.0, 1.0))
        
        results.append({
            "file_path": doc["file_path"],
            "start_line": doc["start_line"],
            "end_line": doc["end_line"],
            "text": doc["text"],
            "language": doc.get("language", "unknown"),
            "scope": doc.get("scope", ""),
            "score": sim
        })
        
    # Sort and return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def query_codebase_batch(
    index_file: str,
    queries: List[str],
    top_k: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Runs multiple queries against the index with a single load.
    Returns a dict mapping each query string to its list of results.
    """
    if not os.path.exists(index_file):
        raise ValueError(f"Index file '{index_file}' does not exist.")
        
    with open(index_file, "rb") as f:
        index_data = pickle.load(f)
    
    version = index_data.get("version", 1)
    if version < INDEX_VERSION:
        raise ValueError(f"Index version {version} is outdated (current: {INDEX_VERSION}). Please re-index.")
    
    config = index_data["config"]
    documents = index_data["documents"]
    
    if not documents:
        return {q: [] for q in queries}
    
    from tq import get_local_embeddings
    query_embs = get_local_embeddings(queries)
    
    engine = TurboQuantex(
        dim=config["dim"],
        bits=config["bits"],
        use_qjl=config["use_qjl"],
        qjl_dim=config["qjl_dim"],
        seed=config["seed"]
    )
    
    # Pre-unpack QJL bits
    doc_qres = []
    for doc in documents:
        q_res = None
        if doc["q_res_packed"] is not None:
            q_res = np.unpackbits(doc["q_res_packed"])[:config["qjl_dim"]].astype(bool)
        doc_qres.append(q_res)
    
    all_results = {}
    for qi, query in enumerate(queries):
        query_emb = query_embs[qi]
        query_norm = np.linalg.norm(query_emb)
        query_norm_u = query_emb / (query_norm + 1e-8) if query_norm > 1e-8 else query_emb
        
        results = []
        for di, doc in enumerate(documents):
            sim = engine.estimate_inner_product(
                doc["norm_x"],
                doc["indices"].astype(np.int32),
                doc_qres[di],
                doc["norm_res"],
                query_norm_u
            )
            sim = float(np.clip(sim, -1.0, 1.0))
            results.append({
                "file_path": doc["file_path"],
                "start_line": doc["start_line"],
                "end_line": doc["end_line"],
                "text": doc["text"],
                "language": doc.get("language", "unknown"),
                "scope": doc.get("scope", ""),
                "score": sim
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        all_results[query] = results[:top_k]
    
    return all_results
