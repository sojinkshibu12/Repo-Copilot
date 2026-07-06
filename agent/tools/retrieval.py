"""Semantic code retrieval tool — bridges the agent to the vector store."""

import logging
from pathlib import Path

from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalToolSet:
    """Tool handlers for semantic codebase search.

    Wraps a VectorStore (backed by Chroma) and provides the handler
    functions that the Orchestrator registers as LLM tool callbacks.
    """

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self._index_root: Path | None = None
        logger.info(
            "RetrievalToolSet initialized (docs_in_store=%d)",
            self.store.count(),
        )

    @property
    def index_root(self) -> Path | None:
        return self._index_root

    @index_root.setter
    def index_root(self, path: str | Path):
        self._index_root = Path(path)

    def semantic_search(self, query: str, k: int = 5) -> str:
        """Search the codebase by semantic meaning.

        Called when the agent needs to find code related to a concept
        that regex/grep can't capture (e.g., 'where is authentication handled').

        Args:
            query: Natural language query about the code.
            k: Number of results to return.

        Returns:
            Formatted string with file paths, line numbers, and code snippets.
        """
        if self.store.count() == 0:
            return (
                "Vector store is empty. The codebase has not been indexed yet. "
                "Run `python scripts/index_codebase.py --repo-path <path>` first. "
                "Falling back to keyword search (grep/glob)."
            )

        results = self.store.search(query, k=k)

        if not results:
            return f"No results found for query: '{query}'. Try a different query or use grep instead."

        lines = [f"Top {len(results)} results for: '{query}'", ""]
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            file_path = meta.get("file", r["id"])
            start = meta.get("start_line", "?")
            end = meta.get("end_line", "?")
            distance = r.get("distance", 0.0)

            lines.append(f"  [{i}] {file_path} (lines {start}-{end}, distance={distance:.3f})")

            snippet = r.get("text", "")
            if snippet:
                # Show first few lines of the snippet
                snippet_lines = snippet.split("\n")[:8]
                for sl in snippet_lines:
                    lines.append(f"       | {sl}")
                if len(snippet.split("\n")) > 8:
                    lines.append("       | ...")

            lines.append("")

        return "\n".join(lines)

    def search_by_file(self, filepath: str, k: int = 3) -> str:
        """Retrieve all indexed chunks for a specific file."""
        results = self.store.search_by_file(filepath, k=k)
        if not results:
            return f"No indexed chunks found for '{filepath}'"

        lines = [f"Found {len(results)} chunk(s) in {filepath}:", ""]
        for r in results:
            meta = r.get("metadata", {})
            start = meta.get("start_line", "?")
            end = meta.get("end_line", "?")
            snippet = r.get("text", "")
            lines.append(f"  Lines {start}-{end}:")
            for sl in snippet.split("\n")[:6]:
                lines.append(f"    | {sl}")
            lines.append("")
        return "\n".join(lines)

    def index_status(self) -> str:
        """Return current index status for agent awareness."""
        count = self.store.count()
        root = self.index_root
        return (
            f"Vector store contains {count} indexed chunks. "
            f"{'Index root: ' + str(root) if root else 'No index root set.'}"
        )

    def get_tool_handlers(self) -> dict:
        """Return dict of tool name → handler for orchestrator registration."""
        return {
            "semantic_search": self.semantic_search,
            "search_by_file": self.search_by_file,
            "index_status": self.index_status,
        }
