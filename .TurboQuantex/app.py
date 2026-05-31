import os
import time
import numpy as np
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from turboquantex import TurboQuantex

app = Flask(__name__, template_folder="templates")

# In-memory database of documents
# Schema:
# {
#    "id": str,
#    "text": str,
#    "original_embedding": np.ndarray (float32),
#    "norm_x": float,
#    "indices": np.ndarray (int32),
#    "q_res": np.ndarray (bool),
#    "norm_res": float
# }
documents_db = []

# Engine config and models cache
config = {
    "bits": 2,
    "use_qjl": True,
    "qjl_dim": 128,
    "seed": 42
}

# Local model cache
local_model = None
turboquantex_engine = None

def get_embedding_dim() -> int:
    return 384

def get_engine() -> TurboQuantex:
    global turboquantex_engine
    dim = get_embedding_dim()
    # Check if we need to initialize or re-initialize the engine
    if (turboquantex_engine is None or 
        turboquantex_engine.dim != dim or 
        turboquantex_engine.bits != config["bits"] or 
        turboquantex_engine.use_qjl != config["use_qjl"] or 
        turboquantex_engine.qjl_dim != config["qjl_dim"] or 
        turboquantex_engine.seed != config["seed"]):
        
        turboquantex_engine = TurboQuantex(
            dim=dim,
            bits=config["bits"],
            use_qjl=config["use_qjl"],
            qjl_dim=config["qjl_dim"],
            seed=config["seed"]
        )
    return turboquantex_engine

def load_local_model():
    global local_model
    if local_model is None:
        # Load local lightweight sentence transformer
        local_model = SentenceTransformer('all-MiniLM-L6-v2')
    return local_model

def get_embedding(text: str) -> np.ndarray:
    """Generates float32 embedding for a given text using local SentenceTransformer."""
    model = load_local_model()
    emb = model.encode(text)
    return np.array(emb, dtype=np.float32)

def recompress_all_documents():
    """Recompresses all indexed original embeddings using the current TurboQuantex configuration."""
    if not documents_db:
        return
    engine = get_engine()
    for doc in documents_db:
        norm_x, indices, q_res, norm_res = engine.compress(doc["original_embedding"])
        doc["norm_x"] = norm_x
        doc["indices"] = indices
        doc["q_res"] = q_res
        doc["norm_res"] = norm_res

def calculate_memory_stats():
    """Calculates theoretical memory usage for original float32 vs TurboQuantex."""
    dim = get_embedding_dim()
    n_docs = len(documents_db)
    
    if n_docs == 0:
        return {
            "count": 0,
            "original_bytes": 0,
            "turbo_bytes": 0,
            "compression_ratio": 0.0,
            "savings_pct": 0.0
        }
        
    # Float32: dim * 4 bytes per vector
    original_bytes = n_docs * dim * 4
    
    # TurboQuantex size per vector:
    # - norm_x: 4 bytes (float32)
    # - norm_res: 4 bytes (float32)
    # - indices: dim * bits (bits)
    # - q_res: qjl_dim (bits)
    bits_per_vector = 32 + 32 + (dim * config["bits"])
    if config["use_qjl"]:
        bits_per_vector += config["qjl_dim"]
        
    bytes_per_vector = int(np.ceil(bits_per_vector / 8))
    turbo_bytes = n_docs * bytes_per_vector
    
    compression_ratio = float(original_bytes) / float(turbo_bytes) if turbo_bytes > 0 else 0.0
    savings_pct = (1.0 - float(turbo_bytes) / float(original_bytes)) * 100.0 if original_bytes > 0 else 0.0
    
    return {
        "count": n_docs,
        "original_bytes": original_bytes,
        "turbo_bytes": turbo_bytes,
        "compression_ratio": round(compression_ratio, 2),
        "savings_pct": round(savings_pct, 2)
    }

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/dashboard")
def index():
    return render_template("index.html")

@app.route("/api/config", methods=["GET", "POST"])
def manage_config():
    global config
    if request.method == "GET":
        return jsonify({"status": "success", "config": config})
    
    # POST - update config
    data = request.get_json() or {}
    
    if "bits" in data:
        config["bits"] = int(data["bits"])
    if "use_qjl" in data:
        config["use_qjl"] = bool(data["use_qjl"])
    if "qjl_dim" in data:
        config["qjl_dim"] = int(data["qjl_dim"])
        
    # Recompress all indexed vectors under new quantizer settings
    recompress_all_documents()
        
    return jsonify({
        "status": "success", 
        "message": "Configuration updated successfully.",
        "config": config
    })

@app.route("/api/index", methods=["POST"])
def index_document():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"status": "error", "message": "Text content cannot be empty."}), 400
        
    try:
        # Generate embedding
        emb = get_embedding(text)
        
        # Compress embedding using TurboQuantex
        engine = get_engine()
        norm_x, indices, q_res, norm_res = engine.compress(emb)
        
        # Add to database
        doc_id = str(len(documents_db) + 1)
        doc = {
            "id": doc_id,
            "text": text,
            "original_embedding": emb,
            "norm_x": norm_x,
            "indices": indices,
            "q_res": q_res,
            "norm_res": norm_res
        }
        documents_db.append(doc)
        
        return jsonify({
            "status": "success",
            "message": "Document indexed successfully.",
            "document": {
                "id": doc_id,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "embedding_dim": len(emb)
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/embed", methods=["POST"])
def embed_texts():
    data = request.get_json() or {}
    texts = data.get("texts", [])
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return jsonify({"status": "error", "message": "Texts list cannot be empty."}), 400
        
    try:
        model = load_local_model()
        embs = model.encode(texts)
        return jsonify({
            "status": "success",
            "embeddings": embs.tolist()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    
    if not query_text:
        return jsonify({"status": "error", "message": "Query text cannot be empty."}), 400
        
    if not documents_db:
        return jsonify({"status": "success", "results": []})
        
    try:
        t0 = time.time()
        # Generate query embedding in full float32 precision
        query_emb = get_embedding(query_text)
        t_embed = time.time() - t0
        
        engine = get_engine()
        
        results_original = []
        results_turbo = []
        
        # Compute similarities
        t_search_orig_start = time.time()
        # 1. Cosine similarity on original float32 embeddings
        query_norm = np.linalg.norm(query_emb)
        for doc in documents_db:
            doc_emb = doc["original_embedding"]
            doc_norm = np.linalg.norm(doc_emb)
            if query_norm > 1e-8 and doc_norm > 1e-8:
                sim = float(np.dot(doc_emb, query_emb) / (doc_norm * query_norm))
            else:
                sim = 0.0
            results_original.append((doc, sim))
        t_search_orig = time.time() - t_search_orig_start
        
        t_search_turbo_start = time.time()
        # 2. Similarity using TurboQuantex inner product estimator
        query_norm_u = query_emb / (query_norm + 1e-8) if query_norm > 1e-8 else query_emb
        
        for doc in documents_db:
            sim_est = engine.estimate_inner_product(
                doc["norm_x"],
                doc["indices"],
                doc["q_res"],
                doc["norm_res"],
                query_norm_u
            )
            # Clip between -1 and 1
            sim_est = float(np.clip(sim_est, -1.0, 1.0))
            results_turbo.append((doc, sim_est))
        t_search_turbo = time.time() - t_search_turbo_start
        
        # Sort results
        results_original.sort(key=lambda x: x[1], reverse=True)
        results_turbo.sort(key=lambda x: x[1], reverse=True)
        
        # Format Top K results for return
        top_orig = results_original[:top_k]
        top_turbo = results_turbo[:top_k]
        
        # Calculate Recall at K
        orig_ids = {d["id"] for d, _ in top_orig}
        turbo_ids = {d["id"] for d, _ in top_turbo}
        matches = orig_ids.intersection(turbo_ids)
        recall = len(matches) / top_k if top_k > 0 else 0.0
        
        # Format final output
        formatted_turbo = []
        for doc, score in top_turbo:
            orig_idx = next((i for i, (d, _) in enumerate(results_original) if d["id"] == doc["id"]), -1)
            orig_score = results_original[orig_idx][1] if orig_idx != -1 else 0.0
            
            formatted_turbo.append({
                "id": doc["id"],
                "text": doc["text"],
                "turbo_score": round(score, 4),
                "original_score": round(orig_score, 4),
                "original_rank": orig_idx + 1
            })
            
        formatted_original = []
        for doc, score in top_orig:
            formatted_original.append({
                "id": doc["id"],
                "text": doc["text"],
                "original_score": round(score, 4)
            })
            
        return jsonify({
            "status": "success",
            "search_time_ms": {
                "embedding": round(t_embed * 1000, 2),
                "original_search": round(t_search_orig * 1000, 4),
                "turbo_search": round(t_search_turbo * 1000, 4)
            },
            "recall_at_k": round(recall, 4),
            "results": formatted_turbo,
            "original_results_preview": formatted_original
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/status", methods=["GET"])
def get_status():
    stats = calculate_memory_stats()
    return jsonify({
        "status": "success",
        "stats": stats
    })

@app.route("/api/reset", methods=["POST"])
def reset_db():
    documents_db.clear()
    return jsonify({"status": "success", "message": "Database reset successfully."})

@app.route("/api/mock_data", methods=["POST"])
def load_mock_data():
    mock_texts = [
        "Artificial Intelligence (AI) is transforming the world, with Large Language Models (LLMs) leading the charge in natural language understanding.",
        "Deep learning models require massive GPU memory, which is why quantization algorithms like TurboQuantex are essential to compress key-value caches.",
        "In vector databases, embeddings represent semantic meaning. High-dimensional vector search enables semantic query matching.",
        "A typical recipe for Italian Pasta Carbonara includes guanciale, egg yolks, pecorino cheese, and freshly cracked black pepper.",
        "Quantum computing harnesses the principles of quantum mechanics, like superposition and entanglement, to perform complex computations.",
        "The Apollo 11 mission landed the first humans on the Moon in 1969, marking a historic achievement in space exploration.",
        "Python is a versatile programming language widely used in data science, web development, and artificial intelligence research.",
        "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states.",
        "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water.",
        "A vector search engine calculates similarity scores between vectors using metrics like Cosine Similarity or Inner Product."
    ]
    
    # Determine how many to index
    data = request.get_json() or {}
    count = int(data.get("count", len(mock_texts)))
    count = min(count, len(mock_texts))
    
    indexed = 0
    errors_list = []
    
    for i in range(count):
        text = mock_texts[i]
        try:
            emb = get_embedding(text)
            engine = get_engine()
            norm_x, indices, q_res, norm_res = engine.compress(emb)
            
            doc_id = str(len(documents_db) + 1)
            doc = {
                "id": doc_id,
                "text": text,
                "original_embedding": emb,
                "norm_x": norm_x,
                "indices": indices,
                "q_res": q_res,
                "norm_res": norm_res
            }
            documents_db.append(doc)
            indexed += 1
        except Exception as e:
            errors_list.append(str(e))
            
    if errors_list and indexed == 0:
        return jsonify({"status": "error", "message": "Failed to load mock data: " + "; ".join(errors_list[:2])}), 500
        
    return jsonify({
        "status": "success",
        "message": f"Successfully indexed {indexed} mock documents.",
        "errors_count": len(errors_list)
    })

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(host="127.0.0.1", port=59402, debug=True)
