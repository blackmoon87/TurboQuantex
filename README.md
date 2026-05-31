# TurboQuantex Codebase Search Engine (100% Offline)

A highly memory-efficient vector search engine for codebase indexing and semantic search. Built on the Google DeepMind/NYU research paper *Online Vector Quantization with Near-optimal Distortion Rate* (ICLR 2026), it compresses vector embeddings (up to **14.2x** RAM reduction) while maintaining dot-product accuracy via a 1-bit **Quantized Johnson-Lindenstrauss (QJL)** residual correction.

This system allows developers and AI coding agents to semantically search massive, multi-file codebases in real-time with negligible memory footprints, avoiding the limitations of LLM context windows and Out-Of-Memory (OOM) errors.

🌐 **Live Landing Page & RAM Simulator**: [https://blackmoon87.github.io/TurboQuantex/](https://blackmoon87.github.io/TurboQuantex/)

All resources are completely isolated inside the `.TurboQuantex/` folder, ensuring no pollution of the developer's main project files or dependencies.

---

## Directory Structure

```
[Your Project Root]
└── .TurboQuantex/
    ├── app.py                  # Flask Daemon (API, Dashboard & File Watcher)
    ├── turboquantex.py         # Core Compression Math (PolarQuant & QJL)
    ├── tq.py                   # CLI: Index, Search, Update, Stats & Batch Search
    ├── turboquantex_skill.py   # Reusable AI Agent Programmatic Skill API
    ├── example_usage.py        # Demo script showing full indexing/updating lifecycle
    ├── setup.bat               # Windows Environment & Dependency Setup Script
    ├── setup.sh                # macOS/Linux Environment & Dependency Setup Script
    ├── index.tq                # Compressed codebase vector index (auto-generated)
    ├── templates/
    │   ├── index.html          # Glassmorphic Web Dashboard
    │   └── landing.html        # Interactive Product Landing Page
    └── example_project/        # Sample codebase directory for demonstration
        ├── app/Http/Controllers/UserController.php
        ├── scripts/data_processor.py
        └── README.md
```

---

## Installation & Setup

### Method A: One-Command AI Deploy (Preferred & Recommended)

If you are working with an AI coding assistant (such as Cursor, VS Code AI, or Antigravity) inside a project, simply type this single prompt:

> **active & use skills : https://github.com/blackmoon87/TurboQuantex**

The AI agent will autonomously:
1. Clone the repository and deploy the `.TurboQuantex` folder into your project root.
2. Install dependencies and configure the environment.
3. Index your codebase and install the post-commit git hook.
4. Start the background daemon with auto file watcher.

### Method B: Manual Terminal Setup (Alternative)

If you prefer to configure the environment manually via the terminal, use the automated setup scripts provided inside the folder:

#### Windows (PowerShell/CMD)
Run:
```powershell
.\.TurboQuantex\setup.bat
```

#### macOS & Linux
Run:
```bash
chmod +x .TurboQuantex/setup.sh
./.TurboQuantex/setup.sh
```

---

## 1. Daemon & File Watcher

Start the Flask daemon to enable three capabilities simultaneously:

```bash
python .TurboQuantex/app.py
```

**What happens when you start the daemon:**

| Capability | Description |
|------------|-------------|
| **Embedding Cache** | The ONNX model stays loaded in RAM — searches drop from ~3s (cold) to <9ms (warm) |
| **File Watcher** | A background thread polls the project directory every 10 seconds. When files change, it waits 5 seconds (debounce) then auto-updates the vector index — **no manual commands needed** |
| **API Server** | REST API on `http://127.0.0.1:59402` for the dashboard, health checks, and programmatic queries |

### Health Check

To verify the daemon and file watcher status:

```bash
curl http://127.0.0.1:59402/api/health
```

Response:
```json
{
  "status": "ok",
  "uptime_seconds": 142.3,
  "model_loaded": true,
  "index_detected": true,
  "file_watcher": {
    "active": true,
    "changes_detected": 5,
    "status": "Successfully updated. Indexed 2 files, removed 0 files."
  }
}
```

---

## 2. CLI Codebase Utility (`tq.py`)

A terminal utility to scan, chunk, index, search, and update large directories recursively.

### A. Index a Codebase Directory

To scan a directory, generate embeddings, compress them, and save the binary database:

```bash
python .TurboQuantex/tq.py index --dir . --index .TurboQuantex/index.tq
```

| Flag | Description |
|------|-------------|
| `--dir` | Directory containing source code to scan |
| `--index` | Path for the output compressed index file (`.tq`) |
| `--bits` | Quantization bits: `2`, `3`, `4`, or `auto` (adaptive) |
| `--use-qjl` | Enable 1-bit QJL residual correction (default: `True`) |
| `--qjl-dim` | QJL sketch dimension: `64`, `128`, or `256` (default: `128`) |
| `--chunk-size` | Maximum characters per code chunk (default: `1200`) |
| `--overlap` | Character overlap between chunks (default: `200`) |
| `--extensions` | Comma-separated extensions filter (e.g., `.py,.php,.js`) |

### B. Search the Index

**Text output (human readable):**

```bash
python .TurboQuantex/tq.py search --index .TurboQuantex/index.tq --query "database insert user record" --top-k 5
```

**JSON output (for AI agents and scripts):**

```bash
python .TurboQuantex/tq.py search --index .TurboQuantex/index.tq --query "database insert user record" --top-k 5 --format json
```

Response:
```json
{
  "status": "success",
  "results": [
    {
      "file_path": "app/Models/User.php",
      "start_line": 45,
      "end_line": 78,
      "score": 0.8432,
      "language": "php",
      "scope": "function insertUser",
      "text": "..."
    }
  ]
}
```

**Filter by programming language:**

```bash
python .TurboQuantex/tq.py search --index .TurboQuantex/index.tq --query "auth logic" --language python --format json
```

### C. Batch Search (Multiple Queries)

Run multiple queries with a single index load — ideal for AI agents exploring a codebase:

```bash
# Comma-separated queries
python .TurboQuantex/tq.py search-batch --index .TurboQuantex/index.tq --queries "auth logic,database connection,file upload" --top-k 3 --format json

# Or from a file (one query per line)
python .TurboQuantex/tq.py search-batch --index .TurboQuantex/index.tq --queries queries.txt --top-k 3 --format json
```

### D. Incremental Update

When code files change, only modified or new files are re-indexed. Unmodified files are loaded instantly from the cache:

```bash
python .TurboQuantex/tq.py update --dir . --index .TurboQuantex/index.tq --format json
```

> **Note:** If the daemon with file watcher is running, updates happen automatically — you don't need this command.

### E. Display Metrics

View index statistics and compression metrics:

```bash
# Human-readable
python .TurboQuantex/tq.py stats --index .TurboQuantex/index.tq

# JSON output
python .TurboQuantex/tq.py stats --index .TurboQuantex/index.tq --format json
```

JSON response:
```json
{
  "file_path": "index.tq",
  "version": 2,
  "model_id": "all-MiniLM-L6-v2",
  "total_chunks": 86,
  "dimensions": 384,
  "bits": 4,
  "compression_ratio": 7.11,
  "savings_percent": 85.94,
  "original_bytes": 132096,
  "compressed_bytes": 18576,
  "disk_bytes": 156287
}
```

### F. Install Git Hook

Register an auto-update git hook so the index updates on every commit:

```bash
python .TurboQuantex/tq.py install-hook
```

---

## 3. Programmatic AI Agent Skill (`turboquantex_skill.py`)

Other scripts or AI agents can import the codebase search skill programmatically:

```python
import sys
sys.path.append('./.TurboQuantex')
from turboquantex_skill import index_codebase, query_codebase, update_codebase, query_codebase_batch

# 1. Full codebase indexing (defaults to adaptive bits)
stats = index_codebase(dir_path=".", index_file=".TurboQuantex/index.tq", bits="auto")
print(f"Compressed RAM Footprint: {stats['disk_size_kb']} KB")

# 2. Semantic query — returns file_path, start_line, end_line, score, language, scope
matches = query_codebase(index_file=".TurboQuantex/index.tq", query="password hashing logic", top_k=3)
for m in matches:
    print(f"{m['file_path']}:{m['start_line']} [{m['language']}] score={m['score']:.4f}")

# 3. Batch query — multiple queries, single index load
batch = query_codebase_batch(
    index_file=".TurboQuantex/index.tq",
    queries=["auth middleware", "database connection", "file upload handler"],
    top_k=3
)
for query, results in batch.items():
    print(f"\n--- {query} ---")
    for r in results:
        print(f"  {r['file_path']}:{r['start_line']} ({r['score']:.4f})")

# 4. Incremental update after modifying files
update_stats = update_codebase(dir_path=".", index_file=".TurboQuantex/index.tq")
print(update_stats['status'])
```

---

## 4. Index Versioning & Compatibility

TurboQuantex uses a versioned index format to prevent silent corruption:

| Field | Description |
|-------|-------------|
| `version` | Index format version (current: `2`). Incremented when the data schema changes |
| `model_id` | Embedding model identifier (`all-MiniLM-L6-v2`). Ensures search results are consistent |

If you update TurboQuantex and the index format has changed, you'll get a clear error message:
```
Error: Index was created with version 1, current version is 2.
Please re-index with: python .TurboQuantex/tq.py index --dir . --index .TurboQuantex/index.tq
```

---

## 5. Chunk Metadata

Every indexed code chunk carries rich metadata:

| Field | Description | Example |
|-------|-------------|---------|
| `file_path` | Relative path to the source file | `app/Models/User.php` |
| `start_line` | First line number of the chunk | `45` |
| `end_line` | Last line number of the chunk | `78` |
| `language` | Auto-detected programming language from file extension | `python`, `php`, `javascript` |
| `scope` | Active function/class context at the chunk boundary | `def process_payment` |
| `score` | Cosine similarity score (0 to 1) | `0.8432` |

Language detection covers 18+ extensions natively. Unknown extensions fall back to `"unknown"`.

---

## 6. Running the Example Codebase

We have packaged a demo script `example_usage.py` that runs through the complete lifecycle. Run the script from the command line:
```bash
python .TurboQuantex/example_usage.py
```
This will:
1. Generate a full compressed index of the `example_project` directory.
2. Query the index for a function.
3. Append a new function to simulate a code modification.
4. Execute an incremental update, demonstrating that it detects the change in milliseconds.
5. Search for the newly added function to prove the index is updated.
6. Automatically clean up and restore the sample codebase.

---

## 7. Real-World Developer Scenarios

### Scenario A: Always-On Daemon (Recommended)

The simplest setup — start the daemon once and forget about it:

1. **Initial Index:**
   ```bash
   python .TurboQuantex/tq.py index --dir . --index .TurboQuantex/index.tq
   ```

2. **Start Daemon:**
   ```bash
   python .TurboQuantex/app.py
   ```

3. **Done.** The file watcher auto-detects changes every 10 seconds. The index stays current without any manual intervention. AI agents and CLI queries use the fast warm model automatically.

### Scenario B: AI Coding Agent Integration (Zero-Prompt)

When working with an AI coding assistant (Cursor, VS Code AI, or Antigravity), the agent automatically discovers and uses TurboQuantex:

1. **Agent Discovery**: The agent reads `.cursorrules` or `important_instruction_4coder_agent.md` at the project root.
2. **Dynamic Querying**: The agent imports `turboquantex_skill.py` and runs queries:
   ```python
   from turboquantex_skill import query_codebase
   results = query_codebase(index_file=".TurboQuantex/index.tq", query="database connection settings")
   ```
3. **Contextual Awareness**: The agent reads matching code and modifies files with full architectural awareness, bypassing LLM context limit issues.

### Scenario C: Git Hook Workflow (Continuous Indexing)

For developers who prefer commit-triggered updates:

1. **Install:**
   ```bash
   python .TurboQuantex/tq.py install-hook
   ```

2. **Code & Commit:**
   ```bash
   git commit -m "feat: add user login endpoint"
   ```

3. **Silent Update:** The hook fires, re-indexes changed files, and prints compression stats in the terminal.

### Scenario D: CI/CD Pipeline Integration

Use JSON output for automated analysis:

```bash
# Check index health in CI
python .TurboQuantex/tq.py stats --index .TurboQuantex/index.tq --format json | jq '.compression_ratio'

# Batch-search for code patterns
python .TurboQuantex/tq.py search-batch --index .TurboQuantex/index.tq --queries "hardcoded password,SQL injection,eval(" --format json
```

---

## 8. Performance Benchmarks

Measured on a standard development machine (no GPU required):

| Metric | Value |
|--------|-------|
| **Compression throughput** | 11,489 vectors/sec |
| **Search throughput** | 19,505 vectors/sec |
| **Per-chunk search latency** | ~53 µs |
| **Warm query (86 chunks)** | 8.9 ms |
| **Cold query (model load + search)** | 342 ms |
| **Batch 5 queries (warm)** | 38.4 ms (7.7 ms/query) |
| **1-file incremental update (warm)** | ~50 ms |
| **Compression ratio (4-bit + QJL)** | 7.11x |
| **RAM savings** | 85.94% |
| **Pearson correlation (fidelity)** | 0.912 |

---

## 9. REST API Endpoints

When the daemon is running on `http://127.0.0.1:59402`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Daemon health, file watcher status, uptime |
| `GET` | `/api/status` | In-memory document count and compression stats |
| `POST` | `/api/local_query` | Search a `.tq` index file via API |
| `POST` | `/api/index` | Index a text document into in-memory store |
| `POST` | `/api/search` | Search in-memory documents |
| `GET` | `/api/config` | Current engine configuration |
| `POST` | `/api/embed` | Generate embedding for text |
| `POST` | `/api/reset` | Clear in-memory document store |
| `GET` | `/` | Web dashboard |
| `GET` | `/dashboard` | Web dashboard (alias) |

### Example: Query a local index via API

```bash
curl -X POST http://127.0.0.1:59402/api/local_query \
  -H "Content-Type: application/json" \
  -d '{"query": "user authentication", "index_file": ".TurboQuantex/index.tq", "top_k": 3}'
```

