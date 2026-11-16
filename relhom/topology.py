"""Topological evaluation through a directed-flag mapping cone.

For a curriculum directed flag complex K and the induced transcript subcomplex
A = K[S], the mapping cone K union_A Cone(A) has the same first homology as the
relative group H_1(K, A). We realize Cone(A) by adding a source vertex "omega"
and an edge omega -> s for every completed course s in S.

Because pyflagser builds the directed flag complex from the adjacency matrix,
all directed 2-simplices involving omega are filled automatically whenever the
required transitive edges exist.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from functools import lru_cache

import networkx as nx
import numpy as np
from pyflagser import flagser_unweighted


@dataclass(frozen=True)
class RelativeH1Components:
    """Exact relative-chain quantities entering beta_0 and beta_1 over GF(2)."""

    relative_vertex_count: int
    relative_edge_count: int
    rank_d1: int
    rank_d2: int
    beta0: int
    beta1: int


@dataclass(frozen=True)
class RelativeH1GainDecomposition:
    """Exact candidate-level decomposition of Delta beta_1."""

    new_vertex_count: int
    absorbed_relative_edges: int
    delta_rank_d1: int
    delta_rank_d2: int
    delta_beta0: int
    delta_beta1: int
    reconstructed_delta_beta1: int


class TopologicalEvaluator:
    """Compute relative first Betti numbers for student transcripts."""

    def __init__(
        self,
        curriculum_graph: nx.DiGraph,
        *,
        ghost_vertex: Hashable = "omega",
        coeff: int = 2,
        validate_dag: bool = True,
    ) -> None:
        if not isinstance(curriculum_graph, nx.DiGraph):
            raise TypeError("curriculum_graph must be a networkx.DiGraph")
        if isinstance(curriculum_graph, nx.MultiDiGraph):
            raise TypeError("MultiDiGraph is not supported; use a simple DiGraph")
        if ghost_vertex in curriculum_graph:
            raise ValueError(
                f"ghost vertex label {ghost_vertex!r} collides with a course node"
            )
        if nx.number_of_selfloops(curriculum_graph):
            raise ValueError("curriculum_graph must not contain self-loops")
        if validate_dag and not nx.is_directed_acyclic_graph(curriculum_graph):
            raise ValueError("curriculum_graph must be a DAG of prerequisites")
        if coeff < 2 or any(
            coeff % divisor == 0
            for divisor in range(2, int(coeff**0.5) + 1)
        ):
            raise ValueError("coeff must be a prime integer >= 2")

        self.curriculum_graph = curriculum_graph.copy()
        self.ghost_vertex = ghost_vertex
        self.coeff = coeff

    def _normalize_transcript(
        self, transcript_nodes: Iterable[Hashable]
    ) -> frozenset[Hashable]:
        transcript = frozenset(transcript_nodes)
        unknown = transcript.difference(self.curriculum_graph.nodes)
        if unknown:
            unknown_display = ", ".join(sorted(map(repr, unknown)))
            raise ValueError(
                f"transcript contains unknown course nodes: {unknown_display}"
            )
        return transcript

    def _build_mapping_cone_graph(
        self, transcript: frozenset[Hashable]
    ) -> nx.DiGraph:
        """Return K with a single cone apex attached to K[S]."""
        augmented = self.curriculum_graph.copy()
        augmented.add_node(self.ghost_vertex)
        augmented.add_edges_from(
            (self.ghost_vertex, node) for node in transcript
        )
        return augmented

    @lru_cache(maxsize=4096)
    def _compute_cached(self, transcript: frozenset[Hashable]) -> int:
        augmented = self._build_mapping_cone_graph(transcript)
        nodelist = list(augmented.nodes)

        adjacency = nx.to_scipy_sparse_array(
            augmented,
            nodelist=nodelist,
            dtype=np.uint8,
            weight=None,
            format="csr",
        )

        result = flagser_unweighted(
            adjacency,
            min_dimension=0,
            max_dimension=2,
            directed=True,
            coeff=self.coeff,
        )

        betti = result.get("betti", [])
        return int(betti[1]) if len(betti) > 1 else 0

    def compute_relative_betti(
        self, transcript_nodes: Iterable[Hashable]
    ) -> int:
        """Return beta_1(K union Cone(K[S])) ~= dim H_1(K, K[S])."""
        transcript = self._normalize_transcript(transcript_nodes)
        return self._compute_cached(transcript)

    def topological_gain(
        self,
        transcript_nodes: Iterable[Hashable],
        candidate: Hashable,
    ) -> int:
        """Return Delta beta_1 = beta_1(S) - beta_1(S union {candidate})."""
        transcript = self._normalize_transcript(transcript_nodes)
        if candidate not in self.curriculum_graph:
            raise ValueError(f"unknown candidate course: {candidate!r}")
        if candidate in transcript:
            return 0

        before = self._compute_cached(transcript)
        after = self._compute_cached(transcript | frozenset({candidate}))
        return before - after


class GF2RelativeHomologyEvaluator:
    """Fast exact H_1(K, K[S]) evaluator over GF(2).

    The directed flag complex is fixed. We precompute its directed edges and
    directed 2-simplices once, then form the relative boundary maps directly
    for each transcript S. Matrix ranks are computed with Python-integer bitset
    Gaussian elimination over GF(2).

    This is mathematically equivalent to the mapping-cone computation for
    beta_1 when coeff=2, but avoids rebuilding a graph and invoking flagser for
    every candidate state.
    """

    def __init__(
        self,
        curriculum_graph: nx.DiGraph,
        *,
        validate_dag: bool = True,
        cache_size: int = 131_072,
    ) -> None:
        if not isinstance(curriculum_graph, nx.DiGraph):
            raise TypeError("curriculum_graph must be a networkx.DiGraph")
        if isinstance(curriculum_graph, nx.MultiDiGraph):
            raise TypeError("MultiDiGraph is not supported; use a simple DiGraph")
        if nx.number_of_selfloops(curriculum_graph):
            raise ValueError("curriculum_graph must not contain self-loops")
        if validate_dag and not nx.is_directed_acyclic_graph(curriculum_graph):
            raise ValueError("curriculum_graph must be a DAG of prerequisites")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")

        self.curriculum_graph = curriculum_graph.copy()
        self.nodes = tuple(self.curriculum_graph.nodes)
        self.node_index = {
            node: index for index, node in enumerate(self.nodes)
        }
        self.edges = tuple(self.curriculum_graph.edges)
        self.edge_index = {
            edge: index for index, edge in enumerate(self.edges)
        }
        self.cache_size = int(cache_size)

        self._edge_vertex_masks: tuple[int, ...] = tuple(
            (1 << self.node_index[source])
            | (1 << self.node_index[target])
            for source, target in self.edges
        )
        self._triangles = self._enumerate_directed_triangles()

        self._cache: OrderedDict[int, int] = OrderedDict()
        self._component_cache: OrderedDict[int, RelativeH1Components] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def _enumerate_directed_triangles(
        self,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Return (vertex_mask, e_uv, e_uw, e_vw) for each 2-simplex."""
        successors = {
            node: set(self.curriculum_graph.successors(node))
            for node in self.nodes
        }
        triangles: list[tuple[int, int, int, int]] = []

        for source, middle in self.edges:
            common_targets = successors[source].intersection(
                successors[middle]
            )
            for target in common_targets:
                edge_sm = self.edge_index[(source, middle)]
                edge_st = self.edge_index[(source, target)]
                edge_mt = self.edge_index[(middle, target)]
                vertex_mask = (
                    (1 << self.node_index[source])
                    | (1 << self.node_index[middle])
                    | (1 << self.node_index[target])
                )
                triangles.append(
                    (
                        vertex_mask,
                        edge_sm,
                        edge_st,
                        edge_mt,
                    )
                )

        return tuple(triangles)

    @staticmethod
    def _gf2_rank(columns: Iterable[int]) -> int:
        """Rank binary column vectors represented as Python integers."""
        basis: dict[int, int] = {}
        rank = 0

        for column in columns:
            value = int(column)
            while value:
                pivot = value.bit_length() - 1
                existing = basis.get(pivot)
                if existing is None:
                    basis[pivot] = value
                    rank += 1
                    break
                value ^= existing

        return rank

    def _normalize_mask(
        self,
        transcript_nodes: Iterable[Hashable],
    ) -> int:
        mask = 0
        unknown: list[Hashable] = []
        for node in set(transcript_nodes):
            index = self.node_index.get(node)
            if index is None:
                unknown.append(node)
            else:
                mask |= 1 << index

        if unknown:
            display = ", ".join(sorted(map(repr, unknown)))
            raise ValueError(
                f"transcript contains unknown course nodes: {display}"
            )
        return mask

    def _compute_components_uncached(
        self,
        transcript_mask: int,
    ) -> RelativeH1Components:
        outside_mask = ((1 << len(self.nodes)) - 1) ^ transcript_mask
        relative_vertex_count = outside_mask.bit_count()

        relative_edge_count = 0
        d1_columns: list[int] = []
        relative_edge_flags = [False] * len(self.edges)

        for edge_index, vertex_mask in enumerate(
            self._edge_vertex_masks
        ):
            boundary = vertex_mask & outside_mask
            if boundary == 0:
                continue
            relative_edge_flags[edge_index] = True
            relative_edge_count += 1
            d1_columns.append(boundary)

        rank_d1 = self._gf2_rank(d1_columns)

        d2_columns: list[int] = []
        for vertex_mask, edge_a, edge_b, edge_c in self._triangles:
            if vertex_mask & outside_mask == 0:
                continue

            boundary_mask = 0
            if relative_edge_flags[edge_a]:
                boundary_mask |= 1 << edge_a
            if relative_edge_flags[edge_b]:
                boundary_mask |= 1 << edge_b
            if relative_edge_flags[edge_c]:
                boundary_mask |= 1 << edge_c

            if boundary_mask:
                d2_columns.append(boundary_mask)

        rank_d2 = self._gf2_rank(d2_columns)
        beta0 = relative_vertex_count - rank_d1
        beta1 = relative_edge_count - rank_d1 - rank_d2

        if beta0 < 0:
            raise RuntimeError(
                "computed negative beta_0; relative boundary rank is "
                "internally inconsistent"
            )
        if beta1 < 0:
            raise RuntimeError(
                "computed negative beta_1; relative boundary ranks are "
                "internally inconsistent"
            )
        return RelativeH1Components(
            relative_vertex_count=relative_vertex_count,
            relative_edge_count=relative_edge_count,
            rank_d1=rank_d1,
            rank_d2=rank_d2,
            beta0=beta0,
            beta1=beta1,
        )

    def _compute_uncached(self, transcript_mask: int) -> int:
        return self._compute_components_uncached(transcript_mask).beta1

    def _compute_components_mask(
        self,
        transcript_mask: int,
    ) -> RelativeH1Components:
        if transcript_mask in self._component_cache:
            value = self._component_cache.pop(transcript_mask)
            self._component_cache[transcript_mask] = value
            return value

        value = self._compute_components_uncached(transcript_mask)
        if len(self._component_cache) >= self.cache_size:
            self._component_cache.popitem(last=False)
        self._component_cache[transcript_mask] = value
        return value

    def _compute_mask(self, transcript_mask: int) -> int:
        if transcript_mask in self._cache:
            self._cache_hits += 1
            value = self._cache.pop(transcript_mask)
            self._cache[transcript_mask] = value
            return value

        self._cache_misses += 1
        value = self._compute_uncached(transcript_mask)

        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)

        self._cache[transcript_mask] = value
        return value

    def compute_relative_betti(
        self,
        transcript_nodes: Iterable[Hashable],
    ) -> int:
        """Return dim H_1(K, K[S]) over GF(2)."""
        return self._compute_mask(
            self._normalize_mask(transcript_nodes)
        )

    def relative_h1_components(
        self,
        transcript_nodes: Iterable[Hashable],
    ) -> RelativeH1Components:
        """Return exact C_1 dimension and boundary ranks for H_1(K,K[S])."""
        return self._compute_components_mask(
            self._normalize_mask(transcript_nodes)
        )

    def relative_h1_gain_decomposition(
        self,
        transcript_nodes: Iterable[Hashable],
        candidate_nodes: Iterable[Hashable] | Hashable,
    ) -> RelativeH1GainDecomposition:
        """Return the exact candidate-volume decomposition of Delta beta_1.

        With q equal to absorbed relative edges and m equal to newly absorbed
        vertices,

            Delta beta_1 = q - m + Delta beta_0 - Delta rank(d_2).

        The equivalent chain-rank form is

            Delta beta_1 = q - Delta rank(d_1) - Delta rank(d_2).
        """
        transcript = set(transcript_nodes)

        try:
            candidate_is_node = candidate_nodes in self.node_index
        except TypeError:
            candidate_is_node = False

        if candidate_is_node:
            candidate_bundle = {candidate_nodes}
        else:
            try:
                candidate_bundle = set(candidate_nodes)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError(
                    "candidate_nodes must be a graph node or iterable of nodes"
                ) from exc

        self._normalize_mask(transcript)
        self._normalize_mask(candidate_bundle)

        before = self.relative_h1_components(transcript)
        after = self.relative_h1_components(
            transcript.union(candidate_bundle)
        )

        new_vertex_count = len(candidate_bundle.difference(transcript))
        absorbed_relative_edges = (
            before.relative_edge_count - after.relative_edge_count
        )
        delta_rank_d1 = before.rank_d1 - after.rank_d1
        delta_rank_d2 = before.rank_d2 - after.rank_d2
        delta_beta0 = before.beta0 - after.beta0
        delta_beta1 = before.beta1 - after.beta1
        reconstructed = (
            absorbed_relative_edges
            - new_vertex_count
            + delta_beta0
            - delta_rank_d2
        )

        if reconstructed != delta_beta1:
            raise RuntimeError(
                "candidate-volume decomposition failed: "
                f"observed={delta_beta1}, reconstructed={reconstructed}"
            )
        if (
            absorbed_relative_edges
            - delta_rank_d1
            - delta_rank_d2
            != delta_beta1
        ):
            raise RuntimeError(
                "chain-rank decomposition failed for candidate gain"
            )

        return RelativeH1GainDecomposition(
            new_vertex_count=new_vertex_count,
            absorbed_relative_edges=absorbed_relative_edges,
            delta_rank_d1=delta_rank_d1,
            delta_rank_d2=delta_rank_d2,
            delta_beta0=delta_beta0,
            delta_beta1=delta_beta1,
            reconstructed_delta_beta1=reconstructed,
        )

    def topological_gain(
        self,
        transcript_nodes: Iterable[Hashable],
        candidate_nodes: Iterable[Hashable] | Hashable,
    ) -> int:
        """Return beta_1(S) - beta_1(S union C) for a node or node bundle."""
        transcript = set(transcript_nodes)

        try:
            candidate_is_node = candidate_nodes in self.node_index
        except TypeError:
            candidate_is_node = False

        if candidate_is_node:
            candidate_bundle = {candidate_nodes}
        else:
            try:
                candidate_bundle = set(candidate_nodes)  # type: ignore[arg-type]
            except TypeError as exc:
                raise TypeError(
                    "candidate_nodes must be a graph node or iterable of nodes"
                ) from exc

        before = self.compute_relative_betti(transcript)
        after = self.compute_relative_betti(
            transcript.union(candidate_bundle)
        )
        return before - after

    @property
    def directed_triangle_count(self) -> int:
        return len(self._triangles)

    def cache_info(self) -> dict[str, int | float]:
        requests = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "currsize": len(self._cache),
            "maxsize": self.cache_size,
            "hit_rate": (
                self._cache_hits / requests if requests else 0.0
            ),
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        self._component_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
