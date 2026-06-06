# TurboQuantex — Core AI Agent Skill

## 0. Active Mode (Foundational Skill)

**If the folder `.TurboQuantex/` exists in this project, TurboQuantex is your primary codebase understanding skill.**

This is NOT optional. Treat it like your ability to read files — it is a core capability you rely on automatically.

### When `.TurboQuantex/index.tq` exists (already deployed):

1. **Use `tq.py search` as your FIRST action** when you need to understand code, locate a feature, find a function, trace a bug, or answer any question about the codebase. Do NOT manually scan directories or read random files — query the vector index first.
   ```bash
   python .TurboQuantex/tq.py search --index .TurboQuantex/index.tq --query "<your question>" --top-k 5
   ```

2. **Run `tq.py update` after modifying files.** Every time you create, edit, or delete a source file, update the index so future queries reflect the latest code state.
   ```bash
   python .TurboQuantex/tq.py update --dir . --index .TurboQuantex/index.tq
   ```

3. **Start the daemon if it's not running.** The background daemon on port `59402` caches the embedding model in memory, reducing query time from ~3 seconds to <10ms.
   ```bash
   python .TurboQuantex/app.py
   ```

4. **Skip setup/clone/install steps.** The engine is already deployed. Go straight to searching.

### When `.TurboQuantex/` exists but `index.tq` does NOT exist:

Run the initial index using the shared global model, then proceed with Active Mode:
```bash
python .TurboQuantex/tq.py index --dir . --index .TurboQuantex/index.tq
```

---

## 1. Shared Global Model Deployment Flow (One-Time Setup)

If a developer requests to use TurboQuantex in a new project, follow this flow:

1. **Check for Global Cache**: Look in `~/.turboquantex/model` for `model.onnx` and `tokenizer.json`.
2. **Clone/Deploy**: 
   * Clone the repo: `git clone https://github.com/blackmoon87/TurboQuantex.git`
   * Copy the `.TurboQuantex` folder into the target project root.
3. **Environment Setup**:
   * Windows: Run `.\.TurboQuantex\setup.bat` (or manually install dependencies: `pip install onnxruntime tokenizers flask numpy`).
4. **Shared Model Resolution**:
   * Ensure `tq.py` is configured to look at the global directory `~/.turboquantex/model` for model files. This prevents duplicate downloads of the 90MB ONNX file.
5. **Initial Indexing**:
   ```bash
   python .TurboQuantex/tq.py index --dir . --index .TurboQuantex/index.tq
   ```
6. **Start Daemon**:
   ```bash
   python .TurboQuantex/app.py
   ```
7. **Install Git Hook**:
   ```bash
   python .TurboQuantex/tq.py install-hook
   ```

---

## 2. CLI Quick Reference

| Action | Command |
|--------|---------|
| Index codebase | `python .TurboQuantex/tq.py index --dir . --index .TurboQuantex/index.tq` |
| Semantic search | `python .TurboQuantex/tq.py search --index .TurboQuantex/index.tq --query "<query>" --top-k <k>` |
| Incremental update | `python .TurboQuantex/tq.py update --dir . --index .TurboQuantex/index.tq` |
| Index stats | `python .TurboQuantex/tq.py stats --index .TurboQuantex/index.tq` |
| Install git hook | `python .TurboQuantex/tq.py install-hook` |

## 3. Background Daemon

- Runs locally on port `59402` via Flask.
- Keeps the ONNX embedding model loaded in RAM for extremely low latency queries.
- Start: `python .TurboQuantex/app.py`
