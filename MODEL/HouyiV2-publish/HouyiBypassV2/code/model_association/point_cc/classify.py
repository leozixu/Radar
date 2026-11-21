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

import numpy as np
import torch


def target_prob_from_regression(
    points_xy,
    points_xy_start,
    points_xy_end,
    threshold
):
    if isinstance(points_xy, torch.Tensor):
        dr = torch.max(
            torch.concatenate(
                [torch.norm(points_xy - points_xy_start, dim=-1, keepdim=True),
                 torch.norm(points_xy - points_xy_end, dim=-1, keepdim=True)],
                dim=-1
            ),
            dim=-1
        ).values
    else:
        dr = np.max(
            np.concatenate(
                [np.linalg.norm(points_xy - points_xy_start, axis=-1, keepdims=True),
                 np.linalg.norm(points_xy - points_xy_end, axis=-1, keepdims=True)],
                axis=-1
            ),
            axis=-1
        )
    return 1 - 2 ** (-dr / threshold)
