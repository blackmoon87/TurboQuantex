import os
import sys
import argparse
import pickle
import time
import numpy as np
from typing import List, Dict, Any, Tuple
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

# Index format version and model identifier
INDEX_VERSION = 2
INDEX_MODEL_ID = "all-MiniLM-L6-v2"

# Language detection map
LANG_MAP = {
    '.py': 'python', '.php': 'php', '.js': 'javascript',
    '.ts': 'typescript', '.go': 'go', '.java': 'java',
    '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp', '.rb': 'ruby',
    '.rs': 'rust', '.swift': 'swift', '.kt': 'kotlin',
    '.sh': 'shell', '.bash': 'shell', '.html': 'html',
    '.css': 'css', '.sql': 'sql', '.md': 'markdown',
}

import urllib.request
import json
import subprocess
import socket

_local_model_cache = None

class ONNXEmbedder:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(script_dir, "model")
        self.model_path = os.path.join(self.model_dir, "model.onnx")
        self.tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        
        self._ensure_model_files()
        
        import onnxruntime as ort
        from tokenizers import Tokenizer
        
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        
        # Load ONNX session using CPUExecutionProvider
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        
    def _ensure_model_files(self):
        os.makedirs(self.model_dir, exist_ok=True)
        
        files = {
            "model.onnx": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
            "tokenizer.json": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
        }
        
        for filename, url in files.items():
            path = os.path.join(self.model_dir, filename)
            if not os.path.exists(path):
                size_str = "~90MB" if filename == "model.onnx" else "~460KB"
                print(f"[*] Downloading local model {filename} ({size_str})...")
                try:
                    import ssl
                    context = ssl._create_unverified_context()
                    with urllib.request.urlopen(url, context=context) as response, open(path, 'wb') as out_file:
                        out_file.write(response.read())
                    print(f"[+] Downloaded {filename} successfully.")
                except Exception as e:
                    print(f"[-] Failed to download {filename}: {e}")
                    raise e
                    
    def encode(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
            
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = [self.tokenizer.encode(t) for t in batch_texts]
            
            # Determine max length in the batch for dynamic padding matching onnx input
            max_len = max(len(e.ids) for e in encoded)
            
            input_ids = []
            attention_mask = []
            token_type_ids = []
            
            for e in encoded:
                ids = e.ids + [0] * (max_len - len(e.ids))
                mask = e.attention_mask + [0] * (max_len - len(e.attention_mask))
                type_ids = e.type_ids + [0] * (max_len - len(e.type_ids))
                
                input_ids.append(ids)
                attention_mask.append(mask)
                token_type_ids.append(type_ids)
                
            input_ids_np = np.array(input_ids, dtype=np.int64)
            attention_mask_np = np.array(attention_mask, dtype=np.int64)
            token_type_ids_np = np.array(token_type_ids, dtype=np.int64)
            
            ort_inputs = {
                "input_ids": input_ids_np,
                "attention_mask": attention_mask_np,
                "token_type_ids": token_type_ids_np
            }
            
            ort_outputs = self.session.run(None, ort_inputs)
            token_embeddings = ort_outputs[0]
            
            # Mean Pooling
            input_mask_expanded = np.expand_dims(attention_mask_np, axis=-1).astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), 1e-9, None)
            
            embeddings = sum_embeddings / sum_mask
            
            # Normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / (norms + 1e-9)
            
            all_embeddings.extend([np.array(e, dtype=np.float32) for e in normalized_embeddings])
            
        return all_embeddings

def is_daemon_running() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(("127.0.0.1", 59402))
            return True
    except Exception:
        return False

def start_daemon_background():
    """Starts the Flask daemon app.py in the background safely and cross-platform."""
    if is_daemon_running():
        return
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    # Locate appropriate python executable (check venv first, otherwise standard sys.executable)
    venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(script_dir, "venv", "bin", "python")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    
    print("[*] Local daemon is not running. Launching Flask service on port 59402 in the background...")
    try:
        # Start daemon process detached from parent CLI process
        if os.name == "nt":
            # On Windows, use CREATE_NO_WINDOW to run completely silently without console popup
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(
                [python_exe, app_path],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=script_dir
            )
        else:
            # On macOS/Linux, run in background
            subprocess.Popen(
                [python_exe, app_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=script_dir,
                start_new_session=True
            )
        
        # Wait up to 10 seconds for the daemon to start and respond
        for _ in range(20):
            if is_daemon_running():
                print("[+] Daemon launched successfully.")
                return
            time.sleep(0.5)
        print("[-] Warning: Background daemon launch timed out, proceeding with local fallback model.")
    except Exception as e:
        print(f"[-] Warning: Failed to start background daemon: {e}")

def get_local_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Queries daemon server at http://127.0.0.1:59402 if running, or falls back to ONNXEmbedder locally."""
    if not is_daemon_running():
        start_daemon_background()
        
    try:
        url = "http://127.0.0.1:59402/api/embed"
        req = urllib.request.Request(
            url, 
            data=json.dumps({"texts": texts}).encode('utf-8'),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        timeout_val = max(5.0, len(texts) * 0.1)
        with urllib.request.urlopen(req, timeout=timeout_val) as response:
            if response.status == 200:
                res_data = json.loads(response.read().decode('utf-8'))
                if res_data.get("status") == "success":
                    return [np.array(e, dtype=np.float32) for e in res_data["embeddings"]]
    except Exception:
        pass # Fail silently and fallback to local model load
        
    global _local_model_cache
    if _local_model_cache is None:
        print("[!] Warning: Daemon unavailable. Using cold local model (slower).", file=sys.stderr)
        _local_model_cache = ONNXEmbedder()
    return _local_model_cache.encode(texts)

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
        detected_language = LANG_MAP.get(ext, "unknown")
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
                    "embedding_text": context_header + chunk_text,
                    "language": detected_language,
                    "scope": active_context
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
                    "embedding_text": context_header + chunk_text,
                    "language": detected_language,
                    "scope": active_context
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
                "embedding_text": context_header + chunk_text,
                "language": detected_language,
                "scope": active_context
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
            "language": chunk.get("language", "unknown"),
            "scope": chunk.get("scope", ""),
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
        "version": INDEX_VERSION,
        "model_id": INDEX_MODEL_ID,
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

def _check_index_version(index_data: Dict[str, Any]):
    """Validates index version compatibility. Exits with clear message if outdated."""
    version = index_data.get("version", 1)
    if version < INDEX_VERSION:
        print(f"Error: Index was created with version {version}, current version is {INDEX_VERSION}. Please re-index with: python .TurboQuantex/tq.py index --dir . --index .TurboQuantex/index.tq")
        sys.exit(1)

def run_update(args):
    use_json = getattr(args, 'format', 'text') == 'json'
    if not use_json:
        print(f"[*] Incrementally updating codebase index: {args.index}")
    from turboquantex_skill import update_codebase
    try:
        res = update_codebase(
            dir_path=args.dir,
            index_file=args.index
        )
        
        if use_json:
            print(json.dumps(res))
        else:
            print(f"[+] Update status: {res['status']}")
            # Load updated index for stats printing
            with open(args.index, "rb") as f:
                index_data = pickle.load(f)
            show_stats_data(index_data, args.index)
    except Exception as e:
        if use_json:
            print(json.dumps({"status": "error", "message": str(e)}))
        else:
            print(f"[-] Error updating codebase: {e}")
        sys.exit(1)

def run_search(args):
    use_json = getattr(args, 'format', 'text') == 'json'
    
    if not os.path.exists(args.index):
        if use_json:
            print(json.dumps({"status": "error", "message": f"Index file '{args.index}' does not exist."}))
        else:
            print(f"Error: Index file '{args.index}' does not exist. Run 'index' command first.")
        sys.exit(1)
        
    # Load index data
    with open(args.index, "rb") as f:
        index_data = pickle.load(f)
    
    _check_index_version(index_data)
        
    config = index_data["config"]
    documents = index_data["documents"]
    
    if not documents:
        if use_json:
            print(json.dumps({"status": "success", "results": []}))
        else:
            print("[-] Index contains no documents.")
        return
    
    # Apply language filter if specified
    lang_filter = getattr(args, 'language', None)
    if lang_filter:
        documents = [doc for doc in documents if doc.get("language", "unknown") == lang_filter.lower()]
        if not documents:
            if use_json:
                print(json.dumps({"status": "success", "results": [], "message": f"No chunks found for language '{lang_filter}'."}))
            else:
                print(f"[-] No chunks found for language '{lang_filter}'.")
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
    
    if use_json:
        json_results = []
        for doc, score in top_results:
            json_results.append({
                "file_path": doc["file_path"],
                "start_line": doc["start_line"],
                "end_line": doc["end_line"],
                "score": round(score, 4),
                "language": doc.get("language", "unknown"),
                "scope": doc.get("scope", ""),
                "text": doc["text"]
            })
        print(json.dumps({"status": "success", "results": json_results}))
    else:
        print(f"\n[+] Top {len(top_results)} matches for: '{args.query}'")
        print("=" * 80)
        for idx, (doc, score) in enumerate(top_results):
            meta_str = f"\nRank #{idx + 1} | Similarity: {score:.4f} | {doc['file_path']} (Lines {doc['start_line']}-{doc['end_line']})"
            print(meta_str.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
            print("-" * 80)
            # Highlight code formatting with indentation
            indented_text = "\n".join("  " + l for l in doc["text"].split("\n")[:15])
            print(indented_text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
            if len(doc["text"].split("\n")) > 15:
                print("  ...")
            print("=" * 80)

def _get_stats_dict(index_data: Dict[str, Any], filepath: str) -> Dict[str, Any]:
    """Computes stats and returns as a dictionary."""
    config = index_data["config"]
    documents = index_data["documents"]
    n_docs = len(documents)
    dim = config["dim"]
    
    disk_bytes = os.path.getsize(filepath)
    original_bytes = n_docs * dim * 4
    
    bits_per_vector = 32 + 32 + (dim * config["bits"])
    if config["use_qjl"]:
        bits_per_vector += config["qjl_dim"]
    bytes_per_vector = int(np.ceil(bits_per_vector / 8))
    theoretical_compressed_bytes = n_docs * bytes_per_vector
    
    ratio = float(original_bytes) / float(theoretical_compressed_bytes) if theoretical_compressed_bytes > 0 else 0.0
    savings = (1.0 - float(theoretical_compressed_bytes) / float(original_bytes)) * 100.0 if original_bytes > 0 else 0.0
    
    return {
        "file_path": filepath,
        "version": index_data.get("version", 1),
        "model_id": index_data.get("model_id", "unknown"),
        "total_chunks": n_docs,
        "dimensions": dim,
        "bits": config["bits"],
        "use_qjl": config["use_qjl"],
        "qjl_dim": config.get("qjl_dim", 0),
        "original_bytes": original_bytes,
        "compressed_bytes": theoretical_compressed_bytes,
        "disk_bytes": disk_bytes,
        "compression_ratio": round(ratio, 2),
        "savings_percent": round(savings, 2)
    }

def show_stats_data(index_data: Dict[str, Any], filepath: str, output_format: str = "text"):
    stats = _get_stats_dict(index_data, filepath)
    
    if output_format == "json":
        print(json.dumps(stats))
        return
    
    config = index_data["config"]
    print("\n" + "=" * 50)
    print("           TURBOQUANTEX INDEX METRICS           ")
    print("=" * 50)
    print(f"File Path:                {stats['file_path']}")
    print(f"Index Version:            {stats['version']}")
    print(f"Total Code Chunks:        {stats['total_chunks']}")
    print(f"Vector Dimensions:        {stats['dimensions']}")
    print(f"Embedding Mode:           local")
    print(f"Quantization Bits:        {stats['bits']} bits")
    print(f"QJL Correction:           {'Active' if stats['use_qjl'] else 'Inactive'}")
    if stats['use_qjl']:
        print(f"QJL Sketch Size:          {stats['qjl_dim']} bits")
    print("-" * 50)
    print(f"Original Float32 RAM:     {stats['original_bytes']:,} Bytes ({stats['original_bytes'] / 1024:.2f} KB)")
    print(f"TurboQuantex Packed RAM:  {stats['compressed_bytes']:,} Bytes ({stats['compressed_bytes'] / 1024:.2f} KB)")
    print(f"Compressed Disk Size:     {stats['disk_bytes']:,} Bytes ({stats['disk_bytes'] / 1024:.2f} KB)")
    print(f"Theoretical RAM Ratio:    {stats['compression_ratio']:.2f}x compression")
    print(f"Theoretical RAM Savings:  {stats['savings_percent']:.2f}% memory reduction")
    print("=" * 50)

def run_stats(args):
    if not os.path.exists(args.index):
        print(f"Error: Index file '{args.index}' does not exist. Run 'index' command first.")
        sys.exit(1)
        
    with open(args.index, "rb") as f:
        index_data = pickle.load(f)
    
    output_format = getattr(args, 'format', 'text')
    show_stats_data(index_data, args.index, output_format)

def run_search_batch(args):
    """Runs multiple queries against the index with a single index load."""
    use_json = getattr(args, 'format', 'text') == 'json'
    
    if not os.path.exists(args.index):
        if use_json:
            print(json.dumps({"status": "error", "message": f"Index file '{args.index}' does not exist."}))
        else:
            print(f"Error: Index file '{args.index}' does not exist.")
        sys.exit(1)
    
    # Load index once
    with open(args.index, "rb") as f:
        index_data = pickle.load(f)
    
    _check_index_version(index_data)
    
    config = index_data["config"]
    documents = index_data["documents"]
    
    # Parse queries: comma-separated string or file path
    if os.path.isfile(args.queries):
        with open(args.queries, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    
    if not queries:
        if use_json:
            print(json.dumps({"status": "error", "message": "No queries provided."}))
        else:
            print("[-] No queries provided.")
        return
    
    # Initialize engine once
    engine = TurboQuantex(
        dim=config["dim"],
        bits=config["bits"],
        use_qjl=config["use_qjl"],
        qjl_dim=config["qjl_dim"],
        seed=config["seed"]
    )
    
    # Batch embed all queries
    query_embs = get_local_embeddings(queries)
    
    # Pre-unpack all document QJL bits
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
            results.append((doc, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:args.top_k]
        
        all_results[query] = [{
            "file_path": doc["file_path"],
            "start_line": doc["start_line"],
            "end_line": doc["end_line"],
            "score": round(score, 4),
            "language": doc.get("language", "unknown"),
            "scope": doc.get("scope", ""),
            "text": doc["text"]
        } for doc, score in top]
    
    if use_json:
        print(json.dumps({"status": "success", "batch_results": all_results}))
    else:
        for query, results in all_results.items():
            print(f"\n[+] Results for: '{query}'")
            print("=" * 80)
            for idx, r in enumerate(results):
                print(f"  Rank #{idx+1} | {r['score']:.4f} | {r['file_path']} (Lines {r['start_line']}-{r['end_line']})")
            print("=" * 80)

def run_install_hook(args):
    # Walk up to find .git directory
    curr = os.path.abspath(args.dir)
    git_dir = None
    for _ in range(5):
        potential = os.path.join(curr, ".git")
        if os.path.isdir(potential):
            git_dir = potential
            break
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
        
    if not git_dir:
        print("Error: Could not find .git folder in the target directory or its parents.")
        sys.exit(1)
        
    hooks_dir = os.path.join(git_dir, "hooks")
    os.makedirs(hooks_dir, exist_ok=True)
    
    hook_path = os.path.join(hooks_dir, "post-commit")
    
    hook_content = """#!/bin/sh
# TurboQuantex post-commit hook
echo "[TurboQuantex] Auto-updating vector index on post-commit..."
python .TurboQuantex/tq.py update --dir . --index .TurboQuantex/index.tq
"""
    
    try:
        # Write/Overwrite hook file
        with open(hook_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(hook_content)
        
        # On POSIX make it executable
        if os.name != "nt":
            import stat
            st = os.stat(hook_path)
            os.chmod(hook_path, st.st_mode | stat.S_IEXEC)
            
        print(f"[+] Git post-commit hook successfully installed at: {hook_path}")
    except Exception as e:
        print(f"[-] Failed to install git hook: {e}")
        sys.exit(1)

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
    parser_update.add_argument("--format", choices=["text", "json"], default="text", help="Output format (text or json)")
 
    # Subparser for search
    parser_search = subparsers.add_parser("search", help="Perform semantic search on indexed codebase")
    parser_search.add_argument("--index", default="codebase_index.tq", help="Path to the index file (.tq)")
    parser_search.add_argument("--query", required=True, help="Query text for semantic search")
    parser_search.add_argument("--top-k", type=int, default=5, help="Number of matching snippets to return")
    parser_search.add_argument("--format", choices=["text", "json"], default="text", help="Output format (text or json)")
    parser_search.add_argument("--language", help="Filter results by language (e.g. python, php, javascript)")
    
    # Subparser for search-batch
    parser_batch = subparsers.add_parser("search-batch", help="Run multiple queries in one index load")
    parser_batch.add_argument("--index", required=True, help="Path to the index file (.tq)")
    parser_batch.add_argument("--queries", required=True, help="Comma-separated queries or path to .txt file with one query per line")
    parser_batch.add_argument("--top-k", type=int, default=3, help="Number of matching snippets per query")
    parser_batch.add_argument("--format", choices=["text", "json"], default="json", help="Output format (text or json)")
    
    # Subparser for stats
    parser_stats = subparsers.add_parser("stats", help="Show index details and compression metrics")
    parser_stats.add_argument("--index", default="codebase_index.tq", help="Path to the index file (.tq)")
    parser_stats.add_argument("--format", choices=["text", "json"], default="text", help="Output format (text or json)")
    
    # Subparser for install-hook
    parser_hook = subparsers.add_parser("install-hook", help="Install git post-commit hook for auto index updating")
    parser_hook.add_argument("--dir", default=".", help="Path to project directory containing .git")

    args = parser.parse_args()
    
    if args.command == "index":
        run_indexing(args)
    elif args.command == "update":
        run_update(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "search-batch":
        run_search_batch(args)
    elif args.command == "stats":
        run_stats(args)
    elif args.command == "install-hook":
        run_install_hook(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
