"""
Graph manager for saving and loading NetworkX graphs.

Also manages the companion SQLite FTS5 index (GraphTextIndex) that provides
BM25-ranked full-text search over graph node names, docstrings, and content.
"""

import gzip
import hashlib
import json
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any

import networkx as nx

from .code_graph.graph_text_index import GraphTextIndex

logger = logging.getLogger(__name__)


def _is_fts5_enabled() -> bool:
    """Check whether FTS5 full-text index feature is enabled.

    Controlled by ``WIKIS_ENABLE_FTS5`` environment variable.
    Default is **ON** — set to ``0`` to explicitly disable.
    When disabled, all consumers gracefully fall back to O(N) graph
    scan when the index is ``None``.
    """
    return os.getenv("WIKIS_ENABLE_FTS5", "1") != "0"


class GraphManager:
    """Manager for persisting and loading NetworkX graphs"""

    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("./data/cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Companion FTS5 index (lazy – created on first use)
        self._fts_index: GraphTextIndex | None = None

    @property
    def fts_index(self) -> GraphTextIndex | None:
        """Lazily-created FTS5 index instance sharing the same cache dir.

        Returns ``None`` when ``WIKIS_ENABLE_FTS5`` is set to ``0``.
        """
        if not _is_fts5_enabled():
            return None
        if self._fts_index is None:
            self._fts_index = GraphTextIndex(cache_dir=str(self.cache_dir))
        return self._fts_index

    def _get_cache_index_path(self) -> Path:
        """Get path to shared cache index file"""
        return self.cache_dir / "cache_index.json"

    def _load_cache_index(self) -> dict[str, Any]:
        """Load cache index"""
        index_path = self._get_cache_index_path()
        if index_path.exists():
            try:
                with open(index_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}

    def _save_cache_index(self, index: dict[str, Any]) -> None:
        """Save cache index (atomic write)."""
        index_path = self._get_cache_index_path()
        try:
            tmp_path = index_path.with_name(f"{index_path.name}.{uuid.uuid4().hex}.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
            os.replace(tmp_path, index_path)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")

    def register_graph_cache(self, repo_identifier: str, cache_key: str, graph_type: str = "combined") -> None:
        """
        Register a graph cache key for a repo_identifier.
        Uses shared cache_index.json with vectorstore.

        Args:
            repo_identifier: Repository identifier (e.g., "owner/repo:branch:commit")
            cache_key: The MD5 hash cache key
            graph_type: Type of graph (combined, import, call)
        """
        index = self._load_cache_index()
        # Store graph keys under a 'graphs' sub-key to avoid collision with vectorstore keys
        if "graphs" not in index:
            index["graphs"] = {}
        graph_key = f"{repo_identifier}:{graph_type}"
        index["graphs"][graph_key] = cache_key

        # If repo_identifier is commit-scoped (repo:branch:commit8), store a canonical pointer.
        try:
            from .repo_resolution import ensure_refs, repo_branch_key, split_repo_identifier

            repo, branch, commit = split_repo_identifier(repo_identifier)
            if repo and branch and commit:
                refs = ensure_refs(index)
                refs[repo_branch_key(repo, branch)] = repo_identifier
        except Exception:  # noqa: S110
            # Never fail graph registration due to pointer bookkeeping.
            pass

        self._save_cache_index(index)
        logger.info(f"Registered graph cache: {graph_key} -> {cache_key}")

    def save_graph(
        self,
        graph: nx.DiGraph,
        repo_path: str,
        graph_type: str = "combined",
        commit_hash: str | None = None,
        repo_identifier: str | None = None,
    ) -> str:
        """
        Save code_graph to compressed pickle file

        Args:
            graph: NetworkX code_graph to save
            repo_path: Repository path for generating cache key
            graph_type: Type of code_graph (combined, import, call)
            commit_hash: Git commit hash for cache key stability
            repo_identifier: Optional repo identifier for cache index registration

        Returns:
            Cache file path
        """
        try:
            # Generate cache key with commit hash
            cache_key = self._generate_cache_key(repo_path, graph_type, commit_hash)
            cache_file = self.cache_dir / f"{cache_key}.code_graph.gz"

            # Save code_graph with compression
            with gzip.open(cache_file, "wb") as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Register in cache index if repo_identifier provided
            if repo_identifier:
                self.register_graph_cache(repo_identifier, cache_key, graph_type)

            logger.info(
                f"Saved {graph_type} code_graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to {cache_file}"
            )

            # Auto-build companion FTS5 index
            try:
                self.build_fts_index(graph, repo_path, graph_type, commit_hash, repo_identifier)
            except Exception as fts_err:
                logger.warning(f"FTS5 index build failed (non-fatal): {fts_err}")

            return str(cache_file)

        except Exception as e:
            logger.error(f"Failed to save code_graph: {e}")
            raise

    def load_graph(
        self, repo_path: str, graph_type: str = "combined", commit_hash: str | None = None
    ) -> nx.DiGraph | None:
        """
        Load code_graph from cache

        Args:
            repo_path: Repository path for generating cache key
            graph_type: Type of code_graph to load
            commit_hash: Git commit hash for cache key lookup

        Returns:
            Loaded NetworkX code_graph or None if not found
        """
        try:
            cache_key = self._generate_cache_key(repo_path, graph_type, commit_hash)
            cache_file = self.cache_dir / f"{cache_key}.code_graph.gz"

            if not cache_file.exists():
                logger.debug(f"Graph cache file not found: {cache_file}")
                return None

            # Load code_graph
            with gzip.open(cache_file, "rb") as f:
                graph = pickle.load(f)  # noqa: S301 — pickle used for graph cache, data is self-generated

            logger.info(
                f"Loaded {graph_type} code_graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges from {cache_file}"
            )

            # Try loading companion FTS5 index (non-fatal if missing)
            try:
                self.fts_index.load(cache_key)
            except Exception:  # noqa: S110
                pass

            return graph

        except Exception as e:
            logger.warning(f"Failed to load code_graph from cache: {e}")
            return None

    def _generate_cache_key(self, repo_path: str, graph_type: str, commit_hash: str | None = None) -> str:
        """Generate cache key for code_graph using commit hash for stability"""
        hasher = hashlib.md5()  # noqa: S324 — content fingerprint for cache key, not cryptographic use
        hasher.update(repo_path.encode())
        hasher.update(graph_type.encode())

        # Use commit hash if available (preferred for reproducibility)
        if commit_hash:
            hasher.update(commit_hash.encode())
        else:
            # Fallback to repo modification time if no commit hash
            if os.path.exists(repo_path):
                try:
                    mtime = os.path.getmtime(repo_path)
                    hasher.update(str(mtime).encode())
                except OSError:
                    pass

        return hasher.hexdigest()

    def graph_exists(self, repo_path: str, graph_type: str = "combined", commit_hash: str | None = None) -> bool:
        """Check if code_graph cache exists"""
        cache_key = self._generate_cache_key(repo_path, graph_type, commit_hash)
        cache_file = self.cache_dir / f"{cache_key}.code_graph.gz"
        return cache_file.exists()

    def load_graph_by_repo_name(self, repo_identifier: str, graph_type: str = "combined") -> nx.DiGraph | None:
        """
        Load graph by repository identifier (convenience method for Ask tool).

        Looks up the cache key from the cache index file.
        Supports both old format (owner/repo:branch) and new format (owner/repo:branch:commit).

        Args:
            repo_identifier: Repository identifier (e.g., "owner/repo:branch" or "owner/repo:branch:commit")
            graph_type: Type of graph to load

        Returns:
            NetworkX graph if found, None otherwise
        """
        index = self._load_cache_index()
        graphs_index = index.get("graphs", {})

        canonical_identifier = repo_identifier
        try:
            from .repo_resolution import resolve_canonical_repo_identifier

            canonical_identifier = resolve_canonical_repo_identifier(
                repo_identifier=repo_identifier,
                cache_dir=self.cache_dir,
                repositories_dir=self.cache_dir / "repositories",
            )
        except ValueError as e:
            logger.warning(str(e))
            return None
        except Exception:
            canonical_identifier = repo_identifier

        graph_key = f"{canonical_identifier}:{graph_type}"
        cache_key = graphs_index.get(graph_key)

        if not cache_key:
            logger.warning(f"No graph cache index entry for {graph_key}")
            # Fallback: try to find any available graph cache
            available_graphs = list(self.cache_dir.glob("*.code_graph.gz"))
            if len(available_graphs) == 1:
                cache_key = available_graphs[0].stem.replace(".code_graph", "")
                logger.info(f"Using single available graph cache: {cache_key}")
            else:
                return None

        cache_file = self.cache_dir / f"{cache_key}.code_graph.gz"
        if not cache_file.exists():
            logger.warning(f"Graph cache file not found: {cache_file}")
            return None

        try:
            logger.info(f"Loading {graph_type} graph from cache key={cache_key} (file={cache_file.name})")
            with gzip.open(cache_file, "rb") as f:
                graph = pickle.load(f)  # noqa: S301 — pickle used for graph cache, data is self-generated
            logger.info(
                f"Loaded {graph_type} graph by repo name: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )

            # Load companion FTS5 index (mirrors load_graph behaviour)
            try:
                if self.fts_index is not None:
                    self.fts_index.load(cache_key)
                    logger.info(f"Loaded companion FTS5 index for cache_key={cache_key}")
            except Exception as fts_err:
                logger.debug(f"FTS5 index not available for {cache_key}: {fts_err}")

            return graph
        except Exception as e:
            logger.warning(f"Failed to load graph by repo name: {e}")
            return None

    def clear_cache(self, repo_path: str = None):
        """Clear code_graph cache"""
        if repo_path:
            # Clear specific repository graphs
            for graph_type in ["combined", "import", "call"]:
                cache_key = self._generate_cache_key(repo_path, graph_type)
                cache_file = self.cache_dir / f"{cache_key}.code_graph.gz"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Removed code_graph cache: {cache_file}")
                # Also remove FTS5 index
                idx = self.fts_index
                if idx is not None:
                    idx.delete(cache_key)
        else:
            # Clear all code_graph cache
            for cache_file in self.cache_dir.glob("*.code_graph.gz"):
                cache_file.unlink()
            for fts_file in self.cache_dir.glob("*.fts5.db*"):
                fts_file.unlink()
            logger.info("Cleared all code_graph and FTS5 cache")

    # ------------------------------------------------------------------
    # FTS5 index management
    # ------------------------------------------------------------------

    def build_fts_index(
        self,
        graph: nx.DiGraph,
        repo_path: str,
        graph_type: str = "combined",
        commit_hash: str | None = None,
        repo_identifier: str | None = None,
    ) -> GraphTextIndex | None:
        """Build a SQLite FTS5 index from the graph and persist to disk.

        Uses the same cache key as the corresponding ``.code_graph.gz`` file
        so the two artefacts are always paired.

        Returns the open :class:`GraphTextIndex` instance, or ``None`` when
        the ``WIKIS_ENABLE_FTS5`` is set to ``0``.
        """
        if not _is_fts5_enabled():
            logger.debug("FTS5 index build skipped (WIKIS_ENABLE_FTS5=0)")
            return None
        cache_key = self._generate_cache_key(repo_path, graph_type, commit_hash)
        idx = self.fts_index
        count = idx.build_from_graph(graph, cache_key)

        if repo_identifier:
            idx.register_in_cache_index(repo_identifier, cache_key)

        logger.info(
            f"Built FTS5 index for {repo_identifier or repo_path}: {count} nodes indexed (cache_key={cache_key})"
        )
        return idx

    def load_fts_index(
        self,
        repo_path: str,
        graph_type: str = "combined",
        commit_hash: str | None = None,
    ) -> GraphTextIndex | None:
        """Load a previously-built FTS5 index by repo path.

        Returns the open index, or ``None`` if no index is cached or
        the ``WIKIS_ENABLE_FTS5`` is set to ``0``.
        """
        if not _is_fts5_enabled():
            return None
        cache_key = self._generate_cache_key(repo_path, graph_type, commit_hash)
        idx = self.fts_index
        if idx is None:
            return None
        if idx.load(cache_key):
            return idx
        return None

    def load_fts_index_by_repo_name(self, repo_identifier: str) -> GraphTextIndex | None:
        """Load FTS5 index by repo identifier (for Ask tool).

        Falls back to the same cache resolution logic as ``load_graph_by_repo_name``.
        Returns ``None`` when ``WIKIS_ENABLE_FTS5`` is set to ``0``.
        """
        if not _is_fts5_enabled():
            return None
        idx = self.fts_index
        if idx is None:
            return None
        if idx.load_by_repo_name(repo_identifier):
            return idx

        # Fallback: try canonical resolution + direct cache key lookup
        index = self._load_cache_index()
        graphs_index = index.get("graphs", {})
        canonical = repo_identifier
        try:
            from .repo_resolution import resolve_canonical_repo_identifier

            canonical = resolve_canonical_repo_identifier(
                repo_identifier=repo_identifier,
                cache_dir=self.cache_dir,
                repositories_dir=self.cache_dir / "repositories",
            )
        except Exception:  # noqa: S110
            pass

        graph_key = f"{canonical}:combined"
        cache_key = graphs_index.get(graph_key)
        if cache_key and idx.load(cache_key):
            return idx

        return None

    def export_graph_data(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Export code_graph to serializable format"""
        return {
            "nodes": [{"id": node_id, **data} for node_id, data in graph.nodes(data=True)],
            "edges": [{"source": source, "target": target, **data} for source, target, data in graph.edges(data=True)],
            "graph_info": {
                "number_of_nodes": graph.number_of_nodes(),
                "number_of_edges": graph.number_of_edges(),
                "is_directed": graph.is_directed(),
                "is_multigraph": graph.is_multigraph(),
            },
        }

    def import_graph_data(self, graph_data: dict[str, Any]) -> nx.DiGraph:
        """Import code_graph from serializable format"""
        graph = nx.DiGraph()

        # Add nodes
        for node_data in graph_data.get("nodes", []):
            node_id = node_data.pop("id")
            graph.add_node(node_id, **node_data)

        # Add edges
        for edge_data in graph_data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            graph.add_edge(source, target, **edge_data)

        return graph

    def analyze_graph(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Analyze code_graph properties"""
        analysis = {
            "basic_stats": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "is_connected": nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph),
            }
        }

        if graph.number_of_nodes() > 0:
            # Degree statistics
            degrees = dict(graph.degree())
            if degrees:
                analysis["degree_stats"] = {
                    "average_degree": sum(degrees.values()) / len(degrees),
                    "max_degree": max(degrees.values()),
                    "min_degree": min(degrees.values()),
                }

            # Try to compute centrality measures for small graphs
            if graph.number_of_nodes() < 1000:
                try:
                    centrality = nx.degree_centrality(graph)
                    analysis["top_central_nodes"] = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                except Exception as e:
                    logger.debug(f"Could not compute centrality: {e}")

            # Component analysis
            if graph.is_directed():
                components = list(nx.weakly_connected_components(graph))
            else:
                components = list(nx.connected_components(graph))

            analysis["components"] = {
                "count": len(components),
                "largest_size": max(len(comp) for comp in components) if components else 0,
                "sizes": [len(comp) for comp in components],
            }

        return analysis

    def find_shortest_path(self, graph: nx.DiGraph, source: str, target: str) -> list | None:
        """Find shortest path between two nodes"""
        try:
            if source in graph and target in graph:
                return nx.shortest_path(graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return None

    def get_node_neighbors(self, graph: nx.DiGraph, node: str, hops: int = 1) -> set:
        """Get neighbors of a node within specified hops"""
        if node not in graph:
            return set()

        neighbors = set()
        current_nodes = {node}

        for _ in range(hops):
            next_nodes = set()
            for n in current_nodes:
                # Get both predecessors and successors
                next_nodes.update(graph.predecessors(n))
                next_nodes.update(graph.successors(n))

            neighbors.update(next_nodes)
            current_nodes = next_nodes - neighbors

        return neighbors - {node}  # Exclude the original node

    def filter_graph_by_file(self, graph: nx.DiGraph, file_path: str) -> nx.DiGraph:
        """Filter code_graph to nodes from specific file"""
        filtered_nodes = [node for node, data in graph.nodes(data=True) if data.get("file_path") == file_path]
        return graph.subgraph(filtered_nodes).copy()

    def filter_graph_by_language(self, graph: nx.DiGraph, language: str) -> nx.DiGraph:
        """Filter code_graph to nodes of specific language"""
        filtered_nodes = [node for node, data in graph.nodes(data=True) if data.get("language") == language]
        return graph.subgraph(filtered_nodes).copy()

    def get_files_in_graph(self, graph: nx.DiGraph) -> set:
        """Get unique file paths in code_graph"""
        files = set()
        for _node, data in graph.nodes(data=True):
            file_path = data.get("file_path")
            if file_path:
                files.add(file_path)
        return files

    def get_languages_in_graph(self, graph: nx.DiGraph) -> set:
        """Get unique languages in code_graph"""
        languages = set()
        for _node, data in graph.nodes(data=True):
            language = data.get("language")
            if language:
                languages.add(language)
        return languages
