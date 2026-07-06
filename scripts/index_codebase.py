#!/usr/bin/env python3
"""Build a vector index of a target repository for semantic search.

Usage:
    python scripts/index_codebase.py --repo-path /path/to/cloned/repo
    python scripts/index_codebase.py --repo-path /path/to/repo --embedder openai
    python scripts/index_codebase.py --repo-path /path/to/repo --embedder local --persist-dir ./data/my-index
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".zsh", ".yml", ".yaml",
    ".json", ".xml", ".md", ".rst", ".html", ".css", ".scss",
    ".sql", ".r", ".m", ".mm",
}

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv",
    "env", ".env", "dist", "build", ".next", ".nuxt",
    "target", "vendor", ".tox", ".eggs", "*.egg-info",
}


def collect_files(
    repo_root: Path,
    extensions: set[str] | None = None,
    ignore_dirs: set[str] | None = None,
) -> list[Path]:
    """Walk a repo and collect all source files."""
    extensions = extensions or SUPPORTED_EXTENSIONS
    ignore_dirs = ignore_dirs or IGNORE_DIRS
    files = []
    for path in repo_root.rglob("*"):
        if path.is_file() and path.suffix in extensions:
            if not any(part.startswith(".") and part != "." for part in path.relative_to(repo_root).parts):
                if not any(ign in path.parts for ign in ignore_dirs):
                    files.append(path)
    return sorted(files)


def chunk_file(
    filepath: Path,
    repo_root: Path,
    max_chars: int = 1500,
    overlap_chars: int = 200,
) -> list[dict]:
    """Split a file into overlapping chunks for embedding."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not text.strip():
        return []

    relative = filepath.relative_to(repo_root)
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_start_line = 1

    for i, line in enumerate(lines, start=1):
        current_chunk.append(line)
        current_size += len(line) + 1
        if current_size >= max_chars:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                "id": f"{relative}:L{chunk_start_line}",
                "text": chunk_text,
                "metadata": {
                    "file": str(relative),
                    "start_line": chunk_start_line,
                    "end_line": i,
                    "suffix": filepath.suffix,
                },
            })
            # overlap: find a good break point near the end
            overlap_text = ""
            overlap_size = 0
            for cl in reversed(current_chunk):
                if overlap_size >= overlap_chars:
                    break
                overlap_text = cl + "\n" + overlap_text
                overlap_size += len(cl) + 1
            current_chunk = overlap_text.strip().split("\n") if overlap_text.strip() else []
            current_size = overlap_size
            chunk_start_line = i - len(current_chunk) + 1

    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        chunks.append({
            "id": f"{relative}:L{chunk_start_line}",
            "text": chunk_text,
            "metadata": {
                "file": str(relative),
                "start_line": chunk_start_line,
                "end_line": len(lines),
                "suffix": filepath.suffix,
            },
        })

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Index a codebase for semantic search")
    parser.add_argument("--repo-path", required=True, help="Path to the cloned repository")
    parser.add_argument("--persist-dir", default=None, help="Chroma persist directory")
    parser.add_argument("--collection", default="codebase", help="Chroma collection name")
    parser.add_argument(
        "--embedder", default=None,
        help="Embedding provider: 'openai', 'local', or 'mock' (default: from env or local)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=1500,
        help="Max characters per chunk",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(SUPPORTED_EXTENSIONS)),
        help="Comma-separated file extensions to include",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear existing collection before indexing",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    if not repo_path.exists():
        logger.error("Repository path does not exist: %s", repo_path)
        sys.exit(1)

    # Collect files
    extensions = set(args.extensions.split(","))
    files = collect_files(repo_path, extensions=extensions)
    logger.info("Found %d files in %s", len(files), repo_path)

    if not files:
        logger.warning("No matching files found. Check --extensions.")
        sys.exit(0)

    # Build chunks
    all_chunks = []
    for filepath in files:
        chunks = chunk_file(filepath, repo_path, args.max_chars)
        all_chunks.extend(chunks)

    logger.info("Generated %d chunks from %d files", len(all_chunks), len(files))

    if not all_chunks:
        logger.warning("No chunks generated (all files empty?).")
        sys.exit(0)

    # Initialise vector store with embedding engine
    from storage.vector_store import VectorStore

    store = VectorStore(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )

    # Configure embedding engine
    if args.embedder:
        os.environ["EMBEDDING_PROVIDER"] = args.embedder

    from core.embeddings import EmbeddingEngine
    engine = EmbeddingEngine()
    store.embedding_engine = engine

    # Clear if requested
    if args.clear:
        store.delete_collection()
        logger.info("Cleared existing collection '%s'", args.collection)

    # Add documents (embeddings computed automatically by the engine)
    store.add_documents(all_chunks)

    logger.info("Indexing complete: %d chunks in '%s'", store.count(), args.collection)


if __name__ == "__main__":
    main()
