"""Spatial gating and bounded sparse assignment for one radar scan."""

import heapq

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, min_weight_full_bipartite_matching
from scipy.spatial import cKDTree


def residual(observed, predicted):
    delta = observed - predicted
    delta[..., 0] = (delta[..., 0] + 0.5) % 1.0 - 0.5
    return delta


def spatial_edges(positions, predictions, widths):
    """Return pairs in the search boxes, without a dense track/point grid."""
    common_width = widths.max(axis=0)
    wrapped_points = positions.copy()
    wrapped_points[:, 0] %= 1.0
    near_zero = np.flatnonzero(wrapped_points[:, 0] <= common_width[0])
    near_one = np.flatnonzero(wrapped_points[:, 0] >= 1.0 - common_width[0])
    offset = np.array([1.0, 0.0, 0.0])
    tree_points = np.concatenate((wrapped_points, wrapped_points[near_zero] + offset,
                                  wrapped_points[near_one] - offset))
    point_ids = np.concatenate((np.arange(len(positions)), near_zero, near_one))
    queries = predictions.copy()
    queries[:, 0] %= 1.0
    tree = cKDTree(tree_points / common_width)
    if len(positions) * len(predictions) > 4_000_000:
        # Bound candidate materialization on anomalously large scans before
        # any feature matrix is allocated. Keep the nearest spatial choices.
        _, candidates = tree.query(queries / common_width, k=96, p=np.inf, distance_upper_bound=1.0)
        rows, ranks = np.where(candidates < len(tree_points))
        return rows, point_ids[candidates[rows, ranks]]
    indexes = tree.query_ball_point(
        queries / common_width, 1.0, p=np.inf, return_sorted=True
    )
    counts = np.fromiter(map(len, indexes), dtype=int, count=len(predictions))
    if not counts.any():
        empty = np.empty(0, dtype=int)
        return empty, empty
    return np.repeat(np.arange(len(predictions)), counts), point_ids[np.concatenate(indexes).astype(int)]


def _bounded_alternatives(cost, best_columns, best_cost, tolerance):
    """Return up to eight assignments within tolerance, using disjoint subproblems."""
    size = len(best_columns)
    serial = 0
    queue = [(best_cost, serial, 0, cost, best_columns)]
    out = []
    while queue and len(out) < 8:
        score, _, start, matrix, columns = heapq.heappop(queue)
        if score > best_cost + tolerance:
            break
        out.append((score - best_cost, columns))
        for split in range(start, size):
            next_matrix = matrix.copy()
            for fixed in range(start, split):
                j = columns[fixed]
                next_matrix[fixed, :] = np.inf
                next_matrix[fixed, j] = matrix[fixed, j]
            next_matrix[split, columns[split]] = np.inf
            try:
                rows, next_columns = linear_sum_assignment(next_matrix)
            except ValueError:  # Forbidding an edge can make this subproblem infeasible.
                continue
            next_cost = next_matrix[rows, next_columns].sum()
            if next_cost <= best_cost + tolerance:
                serial += 1
                heapq.heappush(queue, (next_cost, serial, split, next_matrix, next_columns))
    return out


def assign(row, column, utility, established, track_count, point_count, ambiguity_margin):
    """Return selected edge indices and alternative unreported edges by track.

    Each row also has its own zero-utility miss. Score ties are resolved by
    deterministic row/column order. Alternatives are enumerated only in small
    components containing competing established tracks.
    """
    qualified = np.flatnonzero(utility > 0)
    if not len(qualified):
        return [], {}
    qr, qc = row[qualified], column[qualified]
    graph = coo_matrix((np.ones(len(qr), dtype=np.int8), (qr, qc + track_count)),
                       shape=(track_count + point_count, track_count + point_count)).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    components = labels[qr]
    order = np.argsort(components, kind="stable")
    groups = np.split(qualified[order], np.flatnonzero(np.diff(components[order])) + 1)
    selected = []
    ambiguous = {}
    for edges in groups:
        if len(edges) == 1:
            selected.append(int(edges[0]))
            continue
        rows, local_row = np.unique(row[edges], return_inverse=True)
        columns, local_col = np.unique(column[edges], return_inverse=True)
        track_size, point_size = len(rows), len(columns)
        if track_size == 1 or point_size == 1:
            best_edge = edges[np.argmax(utility[edges])]
            if point_size == 1 and ambiguity_margin > 0 and established[row[best_edge]]:
                possible = edges[established[row[edges]] & (utility[edges] >= utility[best_edge] - ambiguity_margin)]
                if len(possible) > 1:
                    for index in possible:
                        ambiguous[int(row[index])] = {index}
                    continue
            selected.append(int(best_edge))
            continue
        if track_size * (point_size + track_size) > 160_000:
            # A dense matrix is unnecessary even for very large ambiguous components.
            center = utility[edges].max() + 1.0
            local_tracks = np.arange(track_size)
            sparse_cost = coo_matrix((np.r_[center - utility[edges], np.full(track_size, center)],
                                      (np.r_[local_row, local_tracks], np.r_[local_col, point_size + local_tracks])),
                                     shape=(track_size, point_size + track_size)).tocsr()
            matching_rows, matching_cols = min_weight_full_bipartite_matching(sparse_cost)
            pair_to_edge = dict(zip(zip(local_row, local_col), edges))
            selected.extend(int(pair_to_edge[r, c]) for r, c in zip(matching_rows, matching_cols) if c < point_size)
            continue
        cost = np.full((track_size, point_size + track_size), np.inf)
        cost[local_row, local_col] = -utility[edges]
        cost[np.arange(track_size), point_size + np.arange(track_size)] = 0
        index = np.full((track_size, point_size + track_size), -1, dtype=int)
        index[local_row, local_col] = edges
        assignment_rows, best_columns = linear_sum_assignment(cost)
        best_edges = index[assignment_rows, best_columns]
        can_branch = (
            ambiguity_margin > 0 and track_size <= 8 and point_size <= 16
            and np.count_nonzero(established[rows]) >= 2
        )
        if can_branch:
            # Defer only when a different real point can be assigned to at
            # least one of these tracks. A close miss is handled by its dummy.
            for _, possible_columns in _bounded_alternatives(
                cost, best_columns, cost[assignment_rows, best_columns].sum(), ambiguity_margin
            )[1:]:
                possible_edges = index[assignment_rows, possible_columns]
                for r in np.flatnonzero(established[rows] & (possible_edges != best_edges)):
                    if best_edges[r] < 0 and possible_edges[r] < 0:
                        continue
                    choices = ambiguous.setdefault(int(rows[r]), set())
                    choices.update(i for i in (best_edges[r], possible_edges[r]) if i >= 0)
        selected.extend(int(i) for r, i in zip(rows, best_edges) if i >= 0 and r not in ambiguous)
    return selected, ambiguous
