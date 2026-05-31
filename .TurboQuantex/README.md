# TurboQuant Codebase Search Engine (100% Offline)

A highly memory-efficient vector search engine for codebase indexing and semantic search. Built on the Google DeepMind/NYU research paper *Online Vector Quantization with Near-optimal Distortion Rate* (ICLR 2026), it compresses vector embeddings (up to **14.2x** RAM reduction) while maintaining dot-product accuracy via a 1-bit **Quantized Johnson-Lindenstrauss (QJL)** residual correction.

This system allows developers and AI coding agents to semantically search massive, multi-file codebases in real-time with negligible memory footprints, avoiding the limitations of LLM context windows and Out-Of-Memory (OOM) errors.

---

## Directory Structure

```
vector_engine/
├── app.py                  # Flask Web Server (API & Dashboard Backend)
├── turboquant.py           # Core Compression Math (PolarQuant & QJL)
├── turbo_code.py           # CLI Codebase Indexer, Search & Update Utility
├── turboquant_skill.py     # Reusable AI Agent Programmatic Skill API
├── example_usage.py        # Demo script showing full indexing/updating lifecycle
├── setup.bat               # Windows Environment & Dependency Setup Script
├── setup.sh                # macOS/Linux Environment & Dependency Setup Script
├── templates/
│   └── index.html          # Glassmorphic Web Dashboard
└── example_project/        # Sample codebase directory for demonstration
    ├── app/Http/Controllers/UserController.php
    ├── scripts/data_processor.py
    └── README.md
```

---

## Installation & Setup

We provide automated setup scripts to prepare the environment, configure the virtual environment, and install dependencies on all major operating systems.

### Windows (PowerShell/CMD)
Run:
```cmd
setup.bat
```

### macOS & Linux
Run:
```bash
chmod +x setup.sh
./setup.sh
```

---

## 1. Web Dashboard & Daemon Acceleration

Start the Flask server to open the interactive control panel and cache the embedding model in memory:
```bash
python app.py
```
- Open `http://127.0.0.1:5000` in your web browser.
- **Daemon Acceleration**: When this server is running, the CLI and Skill queries route requests through the local daemon, speeding up vector searches from **~3.1 seconds** down to **< 200 milliseconds**!

---

## 2. CLI Codebase Utility (`turbo_code.py`)

A terminal utility is provided to scan, chunk, index, search, and update large directories recursively.

### A. Index a Codebase Directory
To scan a directory, generate embeddings, compress them, and save the binary database (defaults to adaptive `bits="auto"`):
```bash
python turbo_code.py index --dir example_project --index codebase_index.tq
```
- `--dir`: Directory containing source code to scan.
- `--index`: Name/path of the output compressed index file.
- `--bits`: Quantization bits (`2`, `3`, `4`, or `auto` for adaptive bit-rate selection).

### B. Query the Index
To search the codebase semantically:
```bash
python turbo_code.py search --index codebase_index.tq --query "database insert user record" --top-k 2
```

### C. Incremental Update
When code files change, only updated or new files are re-indexed. Unmodified files are loaded instantly from the cache, saving time:
```bash
python turbo_code.py update --dir example_project --index codebase_index.tq
```

### D. Display Footprint Metrics
To view vector statistics and memory-saving percentages:
```bash
python turbo_code.py stats --index codebase_index.tq
```

---

## 3. Programmatic AI Agent Skill (`turboquant_skill.py`)

Other scripts or AI agents (like Antigravity) can import this codebase search skill programmatically:

```python
import sys
sys.path.append('/path/to/vector_engine')
from turboquant_skill import index_codebase, query_codebase, update_codebase

# 1. Full codebase indexing (defaults to adaptive bits)
stats = index_codebase(dir_path="example_project", index_file="db.tq", bits="auto")
print(f"Compressed RAM Footprint: {stats['disk_size_kb']} KB")

# 2. Semantic query
matches = query_codebase(index_file="db.tq", query="password hashing logic", top_k=1)
print(f"Match: {matches[0]['file_path']} - Lines: {matches[0]['start_line']}")

# 3. Incremental update after modifying a file
update_stats = update_codebase(dir_path="example_project", index_file="db.tq")
print(update_stats['status'])
```

---

## 4. Running the Example Codebase

We have packaged a demo script `example_usage.py` that runs through the complete lifecycle. Run the script from the command line:
```bash
python example_usage.py
```
This will:
1. Generate a full compressed index of the `example_project` directory.
2. Query the index for a function.
3. Append a new function to simulate a code modification.
4. Execute an incremental update, demonstrating that it detects the change in milliseconds.
5. Search for the newly added function to prove the index is updated.
6. Automatically clean up and restore the sample codebase.

---

## 5. Real-World Developer Scenarios

### Scenario A: Manual CLI Workflow on a Massive Project

Imagine you are a developer working on a massive project with thousands of files. You want to use the vector engine locally to locate features and search logically without loading all vectors into heavy RAM.

1. **Install within the Project**:
   Open a terminal in your project root and clone/copy the vector engine:
   ```bash
   git clone https://github.com/your-repo/turboquant-vector-engine.git vector_engine
   ./vector_engine/setup.sh
   ```
2. **Execute Full Initial Indexing**:
   Index your codebase. The tool is pre-configured to automatically skip build and dependency directories like `vendor/`, `node_modules/`, `storage/`, and `.git/`:
   ```bash
   python vector_engine/turbo_code.py index --dir . --index codebase_index.tq
   ```
   *Result*: Your entire business logic is indexed and packed into a tiny `codebase_index.tq` binary file.
3. **Daily Semantic Querying**:
   Query your project semantically:
   ```bash
   python vector_engine/turbo_code.py search --index codebase_index.tq --query "user auth login validation" --top-k 1
   ```
4. **Incremental Updating**:
   Whenever files are updated, run:
   ```bash
   python vector_engine/turbo_code.py update --dir . --index codebase_index.tq
   ```
   *Result*: The tool compares timestamps, identifies only changed files, indexes them, and merges them in milliseconds.

---

### Scenario B: Automated Editor & AI Agent Integration (Cursor, Windsurf, Antigravity)

Passing a large codebase directly to an AI coding assistant exceeds context limits and causes high token overhead. By configuring the agent to use the TurboQuant CLI or Skill API, you provide it with a high-precision semantic lookup tool.

#### 1. Custom Rules for Editor Assistants (e.g. `.cursorrules`, `.cursor/rules/`)
You can define custom rules at the root of your workspace to instruct assistants to query the local vector database before suggesting edits:
```markdown
# Codebase Search Protocol
Whenever the user asks to modify a feature or inspect code, locate it first by running:
python vector_engine/turbo_code.py search --index codebase_index.tq --query "<user query description>" --top-k 3

Read the matched code snippets and file paths before proposing any file edits.
```

#### 2. Programmatic Integration for Agents (e.g. Antigravity Skills)
AI Agents can import the skill directly to perform a workspace scan, update the index, and fetch context programmatically:
```python
from vector_engine.turboquant_skill import update_codebase, query_codebase

# 1. Update the index to match the current workspace files
update_codebase(dir_path=".", index_file="codebase_index.tq")

# 2. Query code chunks related to database transactions
matches = query_codebase(index_file="codebase_index.tq", query="db transactions model rollback logic", top_k=3)

# 3. Inject matches into LLM prompt context
prompt_context = "\n\n".join([f"File: {m['file_path']} (Lines {m['start_line']}-{m['end_line']})\n{m['text']}" for m in matches])
```
*Result*: The agent receives a precise code fix with 99% lower token usage and zero memory leaks, running completely locally.
