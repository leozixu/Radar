#  Copyright 2025-2026 Junxiang Xiao
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Union, Sequence

import numpy as np
from route_tools.track_by_corr import AssociatingStatistic


class RegressResidualStatistic:
    """Sum/count accumulator of absolute regression residuals.

    Args:
        total_res: per-coordinate sums of the absolute residuals. E.g.
            ``[x_start_sum, y_start_sum, x_end_sum, y_end_sum,
            x_next_sum, y_next_sum]``.
        count: number of points which contributed to each entry of
            ``total_res``.
            - A scalar means *all* coordinates share the same point count
              (the legacy behavior, which is exactly what it was before
              coordinates could be partially missing).
            - A per-coordinate array (same length as :attr:`total_res`)
              means different coordinates may have been summed over
              different numbers of points. This is needed because some label
              coordinates can be missing for only part of the points (e.g.
              the NEXT label only exists for routes that actually have a next
              point): the sum over the few valid points must be divided by
              their own count, not by the total point count.
            :attr:`avg_res` divides ``total_res`` by :attr:`count` elementwise,
            so it is always the per-coordinate mean.
    """

    def __init__(
        self,
        total_res: Union[float, np.ndarray, Sequence[float]] = np.array(0.),
        count: Union[int, Sequence[int], np.ndarray] = 0
    ):
        self.count = count
        if not isinstance(total_res, np.ndarray):
            total_res = np.array(total_res, dtype=float)
        self.total_res = total_res

    @property
    def avg_res(self):
        # Element-wise mean. A scalar count broadcasts to all coordinates;
        # coordinates whose count is 0 are reported as 0.
        return np.divide(
            self.total_res,
            self.count,
            out=np.zeros_like(self.total_res, dtype=float),
            where=(np.asarray(self.count) != 0)
        )

    def __add__(self, other):
        if not isinstance(other, RegressResidualStatistic):
            raise TypeError(
                f"Cannot add a `RegressResidual` instance with an instance of type `{type(other)}`."
            )
        return RegressResidualStatistic(
            total_res=self.total_res + other.total_res,
            count=self.count + other.count
        )

    def __iadd__(self, other):
        self.count += other.count
        self.total_res = self.total_res + other.total_res
        return self

    def to_dict(self):
        count = (self.count.tolist()
                 if isinstance(self.count, np.ndarray)
                 else int(self.count))
        return {
            "count": count,
            "total_res": self.total_res.tolist(),
            "_dtype": str(self.total_res.dtype)
        }

    @staticmethod
    def from_dict(d: dict) -> "RegressResidualStatistic":
        total_res = np.array(
            d["total_res"], dtype=d["_dtype"]
        )
        count = d["count"]
        if isinstance(count, (list, tuple)):
            count = np.array(count)
        else:
            count = int(count)
        inst = RegressResidualStatistic(
            count=count,
            total_res=total_res
        )
        return inst

    def __eq__(self, other):
        if not isinstance(other, RegressResidualStatistic):
            return False

        if isinstance(self.count, np.ndarray) or isinstance(other.count, np.ndarray):
            counts_equal = np.array_equal(np.asarray(self.count), np.asarray(other.count))
        else:
            counts_equal = self.count == other.count
        return np.allclose(self.total_res, other.total_res, equal_nan=True) and counts_equal

    def __repr__(self):
        """
        >>> stat = RegressResidualStatistic(123, 10)
        >>> eval(repr(stat)) == stat
        True

        >>> stat = RegressResidualStatistic([123, 456], 10)
        >>> eval(repr(stat)) == stat
        True

        >>> stat = RegressResidualStatistic(np.array([123, 456]), 10)
        >>> eval(repr(stat)) == stat
        True

        >>> stat = RegressResidualStatistic(np.array([123, 456]), np.array([10, 12]))
        >>> eval(repr(stat)) == stat
        True

        """
        count_repr = (f"np.array({self.count.tolist()})"
                      if isinstance(self.count, np.ndarray)
                      else repr(self.count))
        return f"RegressResidualStatistic(total_res={self.total_res.tolist()}, count={count_repr})"


@dataclasses.dataclass
class AccuracyStatistic(AssociatingStatistic):
    correct: int = 0
    total: int = 0

    def to_dict(self) -> dict:
        return {"correct": int(self.correct), "total": int(self.total)}

    @staticmethod
    def from_dict(d: dict) -> "AccuracyStatistic":
        return AccuracyStatistic(correct=d["correct"], total=d["total"])
