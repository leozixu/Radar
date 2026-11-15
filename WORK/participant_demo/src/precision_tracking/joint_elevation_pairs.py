"""Jointly assign returns for tracks with a stable relative elevation."""

from collections import Counter
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .pair_swap_probe import candidate_features
from .relative_elevation_assoc import fit_relative_elevation, recent_elevation, reliable_anchors


@dataclass(frozen=True, slots=True)
class PairOption:
    first: int
    second: int
    utility: float


@dataclass(frozen=True, slots=True)
class PairGroup:
    first_row: int
    second_row: int
    options: tuple[PairOption, ...]


def stable_pairs(tracks, circle, spread_limit, available_rows, minimum_history=5):
    """Identify unambiguous track pairs using only prior common-scan measurements."""
    eligible = np.asarray([len(track.history) >= minimum_history for track in tracks]) & available_rows
    indices = np.flatnonzero(eligible)
    if len(indices) < 2:
        return []
    history = recent_elevation([tracks[i] for i in indices], circle)
    _, individual_spread, _, _ = fit_relative_elevation(history, np.zeros((1, history.shape[1])))
    # Very smooth independent objects can have a fixed offset accidentally. A
    # shared, non-linear elevation fluctuation is much stronger pair evidence.
    dynamic = individual_spread[:, 0] >= .002
    history, indices = history[dynamic], indices[dynamic]
    if len(indices) < 2:
        return []
    predicted, spreads, counts, recent = fit_relative_elevation(history, history)
    related = (counts >= minimum_history) & recent & (spreads <= spread_limit)
    np.fill_diagonal(related, False)
    unique = related.sum(axis=1) == 1
    left, right = np.nonzero(np.triu(related & unique[:, None] & unique[None, :], k=1))
    return [(int(indices[i]), int(indices[j]), float(predicted[i, j])) for i, j in zip(left, right)]


def strong_columns(rows, cols, probability, n_tracks, n_points):
    best_row = np.zeros(n_tracks)
    best_col = np.zeros(n_points)
    np.maximum.at(best_row, rows, probability)
    np.maximum.at(best_col, cols, probability)
    mutual = np.flatnonzero((probability == best_row[rows]) & (probability == best_col[cols]))
    reliable = reliable_anchors(rows, cols, probability, mutual, .80, .12, n_tracks, n_points)
    owner = np.full(n_points, -1)
    owner[cols[reliable]] = rows[reliable]
    return owner, set(reliable.tolist())


def candidate_groups(pairs, rows, cols, probability, utility, positions, n_tracks, *, tolerance, bonus,
                     min_probability):
    owners, reliable = strong_columns(rows, cols, probability, n_tracks, len(positions))
    ranked_by_row = np.lexsort((-probability, rows))
    boundaries = np.cumsum(np.bincount(rows, minlength=n_tracks))
    starts = np.r_[0, boundaries[:-1]]
    groups = []
    for row_a, row_b, difference in pairs:
        proposals = []
        for row in (row_a, row_b):
            edges = ranked_by_row[starts[row]:boundaries[row]]
            owner = owners[cols[edges]]
            allowed = (probability[edges] >= min_probability) & ((owner < 0) | (owner == row_a) | (owner == row_b))
            proposals.append(edges[allowed][:8])
        a, b = proposals
        if not len(a) or not len(b):
            continue
        observed = positions[cols[a], 1, None] - positions[cols[b], 1]
        error = np.abs(observed - difference) / tolerance
        score = utility[a, None] + utility[b] + bonus * (1 - .5 * error**2)
        valid = (cols[a, None] != cols[b]) & (error <= 1.) & (score > 0)
        aa, bb = np.nonzero(valid)
        if not len(aa):
            continue
        options = sorted((PairOption(int(a[i]), int(b[j]), float(score[i, j])) for i, j in zip(aa, bb)),
                         key=lambda option: (-option.utility, option.first, option.second))
        if options[0].first in reliable and options[0].second in reliable:
            # Both columns independently choose this exact allocation already.
            continue
        groups.append(PairGroup(row_a, row_b, tuple(options[:3])))
    return groups


def optimize_component(groups, rows, cols, utility, component_rows, component_cols, minimum_gain):
    """Optimize a bounded set of pair choices jointly with all remaining individual edges."""
    n_rows, n_cols = len(component_rows), len(component_cols)
    costs = np.full((n_rows, n_cols + n_rows), np.inf)
    valid = utility > 0
    costs[np.searchsorted(component_rows, rows[valid]), np.searchsorted(component_cols, cols[valid])] = -utility[valid]
    costs[np.arange(n_rows), n_cols + np.arange(n_rows)] = 0.
    rr, cc = linear_sum_assignment(costs)
    best_score = -costs[rr, cc].sum() + minimum_gain
    chosen = ()
    alternatives = list(product(*((None, *group.options) for group in groups)))
    # Explore high potential alternatives first, preserving a clear no-pair
    # fallback when competition makes a pair's nominal bonus misleading.
    alternatives.sort(key=lambda items: -sum(option.utility for option in items if option is not None))
    for combination in alternatives:
        pairs = tuple(option for option in combination if option is not None)
        if not pairs:
            continue
        indices = tuple(edge for option in pairs for edge in (option.first, option.second))
        occupied = cols[list(indices)]
        if len(np.unique(occupied)) != len(occupied):
            continue
        row_mask = np.ones(n_rows, dtype=bool)
        col_mask = np.ones(n_cols + n_rows, dtype=bool)
        row_mask[np.searchsorted(component_rows, rows[list(indices)])] = False
        col_mask[np.searchsorted(component_cols, occupied)] = False
        remaining = costs[np.ix_(row_mask, col_mask)]
        rr, cc = linear_sum_assignment(remaining)
        score = sum(option.utility for option in pairs) - remaining[rr, cc].sum()
        if score > best_score:
            chosen, best_score = indices, score
    return chosen


def select_joint_options(groups, rows, cols, utility, n_tracks, n_points, minimum_gain, max_groups):
    if not groups:
        return np.empty(0, dtype=int)
    group_edges = np.array([edge for group in groups for option in group.options
                            for edge in (option.first, option.second)], dtype=int)
    relevant = (utility > 0)
    relevant[group_edges] = True
    graph_row = np.r_[rows[relevant], [group.first_row for group in groups]]
    graph_col = np.r_[cols[relevant] + n_tracks, [group.second_row for group in groups]]
    graph = coo_matrix((np.ones(len(graph_row), dtype=np.int8), (graph_row, graph_col)),
                       shape=(n_tracks + n_points, n_tracks + n_points)).tocsr()
    _, labels = connected_components(graph, directed=False)
    grouped = {}
    for group in groups:
        grouped.setdefault(labels[group.first_row], []).append(group)
    chosen = []
    for label, options in grouped.items():
        active_rows = np.flatnonzero(labels[:n_tracks] == label)
        active_cols = np.flatnonzero(labels[n_tracks:] == label)
        active = np.flatnonzero(labels[rows] == label)
        # The working edge indices remain global. Inactive edges get -inf,
        # keeping the option indices stable across components.
        component_utility = np.full(len(utility), -np.inf)
        component_utility[active] = utility[active]
        bounded = sorted(options, key=lambda group: -group.options[0].utility)[:max_groups]
        chosen.extend(optimize_component(bounded, rows, cols, component_utility, active_rows, active_cols,
                                         minimum_gain))
    return np.asarray(chosen, dtype=int)


class JointElevationMatcher:
    def __init__(self, base_solver, *, spread=.0005, tolerance=.0009, bonus=1.4,
                 min_probability=.065, minimum_gain=.25, max_groups=3, minimum_history=5):
        self.base = base_solver
        self.spread = spread
        self.tolerance = tolerance
        self.bonus = bonus
        self.min_probability = min_probability
        self.minimum_gain = minimum_gain
        self.max_groups = max_groups
        self.minimum_history = minimum_history
        self.statistics = Counter()

    def associate(self, tracks, positions, amplitudes, circle, model, floor, mix, ambiguity):
        if not tracks or not len(positions):
            return [], np.empty(0, dtype=int)
        rows, cols, features = candidate_features(tracks, positions, amplitudes, circle)
        if not len(rows):
            return [], np.empty(0, dtype=int)
        probability = model.predict(features, num_threads=1)
        utility = np.log(np.clip(probability, 1e-6, 1.)) - np.log(floor) - mix * features[:, 10]
        available = np.bincount(rows[probability >= self.min_probability], minlength=len(tracks)) > 0
        pairs = stable_pairs(tracks, circle, self.spread, available, self.minimum_history)
        self.statistics["stable_pairs"] += len(pairs)
        groups = candidate_groups(pairs, rows, cols, probability, utility, positions, len(tracks),
                                  tolerance=self.tolerance, bonus=self.bonus, min_probability=self.min_probability)
        self.statistics["proposed_groups"] += len(groups)
        chosen = select_joint_options(groups, rows, cols, utility, len(tracks), len(positions),
                                      self.minimum_gain, self.max_groups)
        self.statistics["selected_pairs"] += len(chosen) // 2
        self.statistics["selected_below_floor"] += int((utility[chosen] <= 0).sum())
        if len(chosen):
            locked_rows = np.zeros(len(tracks), dtype=bool)
            locked_cols = np.zeros(len(positions), dtype=bool)
            locked_rows[rows[chosen]] = True
            locked_cols[cols[chosen]] = True
            remaining = ~locked_rows[rows] & ~locked_cols[cols]
            edge_data = rows[remaining], cols[remaining], features[remaining], probability[remaining]
        else:
            edge_data = rows, cols, features, probability
        matches, held = self.base.associate(tracks, positions, amplitudes, circle, model, floor, mix, ambiguity,
                                            edge_data=edge_data)
        paired = [(int(rows[edge]), int(cols[edge]), float(max(.60, probability[edge]))) for edge in chosen]
        return matches + paired, held
