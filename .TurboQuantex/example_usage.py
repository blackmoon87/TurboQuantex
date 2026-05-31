"""
example_usage.py - Demonstration of the TurboQuantex Programmatic Skill API

This script demonstrates:
1. Indexing a directory recursively ('example_project').
2. Performing semantic search queries on the compressed database.
3. Dynamically modifying a file and performing an incremental update (re-indexing ONLY changed files).
4. Running search query again to verify updated codebase.
"""

import os
import sys
import time

# Ensure the vector_engine directory is in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import turboquantex_skill

INDEX_FILE = "demo_index.tq"
DIR_TO_INDEX = "example_project"

def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f" {title.upper()} ")
    print("=" * 60)

def main():
    print_separator("1. Full Codebase Indexing")
    print(f"[*] Scanning and indexing directory: '{DIR_TO_INDEX}'...")
    
    # Run full indexing using local sentence transformers (bits="auto" selects 4 bits for small projects)
    stats = turboquantex_skill.index_codebase(
        dir_path=DIR_TO_INDEX,
        index_file=INDEX_FILE,
        bits="auto"
    )
    
    print("[+] Indexing complete!")
    print(f"    - Status:       {stats['status']}")
    print(f"    - Total Chunks: {stats['chunks']}")
    print(f"    - Dimensions:   {stats['dimensions']}")
    print(f"    - Packed Size:  {stats['disk_size_kb']} KB")
    
    # Query 1
    print_separator("2. Semantic Search Query")
    query = "hashing user passwords register database insert"
    print(f"[*] Searching for: '{query}'")
    
    results = turboquantex_skill.query_codebase(INDEX_FILE, query, top_k=1)
    if results:
        match = results[0]
        print(f"[+] Top Match (Similarity Score: {match['score']:.4f}):")
        print(f"    File: {match['file_path']} (Lines {match['start_line']}-{match['end_line']})")
        print("-" * 60)
        print(match['text'].strip())
    else:
        print("[-] No matches found.")

    # Simulating a file modification
    print_separator("3. Incremental Update Demonstration")
    file_to_modify = os.path.join(DIR_TO_INDEX, "scripts", "data_processor.py")
    print(f"[*] Simulating modification in '{file_to_modify}'...")
    
    # Append a new utility method to data_processor.py
    with open(file_to_modify, "a", encoding="utf-8") as f:
        f.write("\n\n    def calculate_sum(self, values):\n        \"\"\"New method: Compute the sum of list values.\"\"\"\n        if not values:\n            return 0.0\n        return sum(values)\n")
        
    print("[*] Running incremental update...")
    t_start = time.time()
    update_stats = turboquantex_skill.update_codebase(
        dir_path=DIR_TO_INDEX,
        index_file=INDEX_FILE
    )
    t_duration = time.time() - t_start
    
    print(f"[+] Incremental update completed in {t_duration:.4f} seconds!")
    print(f"    - Status:         {update_stats['status']}")
    print(f"    - Updated Files:  {update_stats.get('added_files', 0)}")
    print(f"    - Chunks Count:   {update_stats['chunks']}")

    # Query 2 (Search for the newly added method)
    print_separator("4. Searching Updated Codebase")
    query_new = "calculate sum list values method"
    print(f"[*] Searching for: '{query_new}'")
    
    results_new = turboquantex_skill.query_codebase(INDEX_FILE, query_new, top_k=1)
    if results_new:
        match = results_new[0]
        print(f"[+] Top Match (Similarity Score: {match['score']:.4f}):")
        print(f"    File: {match['file_path']} (Lines {match['start_line']}-{match['end_line']})")
        print("-" * 60)
        print(match['text'].strip())
    else:
        print("[-] No matches found.")

    # Restoring data_processor.py to original state (cleaning up modification)
    print_separator("5. Cleanup and Restore")
    print("[*] Restoring data_processor.py to original state...")
    try:
        with open(file_to_modify, "r", encoding="utf-8") as f:
            content = f.read()
        marker = "\n\n    def calculate_sum"
        if marker in content:
            clean_content = content.split(marker)[0]
            with open(file_to_modify, "w", encoding="utf-8") as f:
                f.write(clean_content)
        print("[+] data_processor.py restored.")
        # Clean up index file
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        print("[+] demo_index.tq index file removed.")
    except Exception as e:
        print(f"Warning: Failed to restore data_processor.py: {e}")

if __name__ == "__main__":
    main()
