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
    ├── app.py                  # Flask Daemon (API & File Watcher)
    ├── turboquantex.py         # Core Compression Math (PolarQuant & QJL)
    ├── tq.py                   # CLI Codebase Indexer, Search & Update Utility
    ├── turboquantex_skill.py   # Reusable AI Agent Programmatic Skill API
    ├── example_usage.py        # Demo script showing full indexing/updating lifecycle
    ├── setup.bat               # Windows Environment & Dependency Setup Script
    ├── setup.sh                # macOS/Linux Environment & Dependency Setup Script
    └── example_project/        # Sample codebase directory for demonstration
        ├── app/Http/Controllers/UserController.php
        ├── scripts/data_processor.py
        └── README.md
```

---

## Installation & Setup

### Method A: Zero-Prompt AI Agent Autopilot (Preferred & Recommended)

If you are working with an AI coding assistant (such as Cursor, VS Code AI, or Antigravity) inside a project, you do not need to download or install this tool manually. Simply type this prompt to your AI assistant:
> "Deploy and start using TurboQuantex from https://github.com/blackmoon87/TurboQuantex"

The AI agent will autonomously:
1. Clone the repository `https://github.com/blackmoon87/TurboQuantex.git` in the background.
2. Deploy the isolated `.TurboQuantex` engine subfolder directly into your current project root.
3. Configure the virtual environment (`.\.TurboQuantex\setup.bat` or `setup.sh`) and install requirements.
4. Index your codebase, install the post-commit git hook, and spin up the silent background daemon.
5. Provide you with instant semantic search and query capabilities.

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

## 1. Daemon Acceleration & File Watcher

Start the Flask daemon to enable background embedding caching and automatic file watching:
```bash
python .TurboQuantex/app.py
```
- **Daemon Acceleration**: When this server is running on the custom private port `59402`, the CLI and Skill queries route requests through the local daemon, speeding up vector searches from **~3.1 seconds** down to **< 200 milliseconds**!

---

## 2. CLI Codebase Utility (`tq.py`)

A terminal utility is provided to scan, chunk, index, search, and update large directories recursively.

### A. Index a Codebase Directory
To scan a directory, generate embeddings, compress them, and save the binary database:
```bash
python .TurboQuantex/tq.py index --dir example_project --index .TurboQuantex/codebase_index.tq
```
- `--dir`: Directory containing source code to scan.
- `--index`: Name/path of the output compressed index file.
- `--bits`: Quantization bits (`2`, `3`, `4`, or `auto` for adaptive bit-rate selection).

### B. Query the Index
To search the codebase semantically:
```bash
python .TurboQuantex/tq.py search --index .TurboQuantex/codebase_index.tq --query "database insert user record" --top-k 2
```

### C. Incremental Update
When code files change, only updated or new files are re-indexed. Unmodified files are loaded instantly from the cache, saving time:
```bash
python .TurboQuantex/tq.py update --dir example_project --index .TurboQuantex/codebase_index.tq
```

### D. Display Footprint Metrics
To view vector statistics and memory-saving percentages:
```bash
python .TurboQuantex/tq.py stats --index .TurboQuantex/codebase_index.tq
```

---

## 3. Programmatic AI Agent Skill (`turboquantex_skill.py`)

Other scripts or AI agents (like Antigravity) can import this codebase search skill programmatically:

```python
import sys
sys.path.append('./.TurboQuantex')
from turboquantex_skill import index_codebase, query_codebase, update_codebase

# 1. Full codebase indexing (defaults to adaptive bits)
stats = index_codebase(dir_path="example_project", index_file=".TurboQuantex/db.tq", bits="auto")
print(f"Compressed RAM Footprint: {stats['disk_size_kb']} KB")

# 2. Semantic query
matches = query_codebase(index_file=".TurboQuantex/db.tq", query="password hashing logic", top_k=1)
print(f"Match: {matches[0]['file_path']} - Lines: {matches[0]['start_line']}")

# 3. Incremental update after modifying a file
update_stats = update_codebase(dir_path="example_project", index_file=".TurboQuantex/db.tq")
print(update_stats['status'])
```

---

## 4. Running the Example Codebase

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

## 5. Real-World Developer Scenarios

### Scenario A: Manual CLI Workflow on a Massive Project

Imagine you are a developer working on a massive project with thousands of files. You want to use the vector engine locally to locate features and search logically without loading all vectors into heavy RAM.

1. **Install within the Project**:
   Copy the `.TurboQuantex` folder to your project root.
   ```bash
   .TurboQuantex/setup.sh
   ```
2. **Execute Full Initial Indexing**:
   Index your codebase. The tool is pre-configured to automatically skip build and dependency directories like `vendor/`, `node_modules/`, `storage/`, `.git/`, and `.TurboQuantex/` itself:
   ```bash
   python .TurboQuantex/tq.py index --dir . --index .TurboQuantex/codebase_index.tq
   ```
   *Result*: Your entire business logic is indexed and packed into a tiny `codebase_index.tq` binary file.
3. **Daily Semantic Querying**:
   Query your project semantically:
   ```bash
   python .TurboQuantex/tq.py search --index .TurboQuantex/codebase_index.tq --query "user auth login validation" --top-k 1
   ```
4. **Incremental Updating**:
   Whenever files are updated, run:
   ```bash
   python .TurboQuantex/tq.py update --dir . --index .TurboQuantex/codebase_index.tq
   ```
   *Result*: The tool compares timestamps, identifies only changed files, indexes them, and merges them in milliseconds.

### Scenario B: AI Coding Agent Integration (Zero-Prompt Coding)

When working with an AI coding assistant (like Cursor, VS Code AI, or Antigravity), you want the agent to automatically search your codebase semantically to answer questions and write code without you manually prompting it to index files.

1. **Agent Discovery**: The coding agent reads `.cursorrules` or detects `important_instruction_4coder_agent.md` at your project root.
2. **Dynamic Querying**: The agent dynamically imports `turboquantex_skill.py` and runs queries in the background:
   ```python
   from turboquantex_skill import query_codebase
   results = query_codebase(index_file=".TurboQuantex/index.tq", query="database connection settings")
   ```
3. **Contextual Awareness**: The agent reads the matching lines and modifies files with full architectural awareness, bypassing LLM context limit issues.

### Scenario C: Automated Git Hook Workflow (Continuous Local Indexing)

To make sure your vector index is always up-to-date with your latest git branch changes without running manual commands:

1. **Install Git Hook**:
   Register the post-commit git hook in your local repository:
   ```bash
   python .TurboQuantex/tq.py install-hook
   ```
2. **Code & Commit**:
   Write code normally and run `git commit -m "feat: add user login endpoint"`.
3. **Silent Background Update**:
   The git hook automatically fires, identifies modified files, updates the local vector database `.TurboQuantex/index.tq` in milliseconds, and prints the updated compression stats directly in your terminal.

