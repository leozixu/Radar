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

from typing import Union, Literal


class ModelParts:
    def __init__(self):
        self._backbone = {}
        self._classify = {}
        self._regress = {}

    def add_backbone_comps(self, comps):
        self._backbone.update(comps)

    def add_classify_comps(self, comps):
        self._classify.update(comps)

    def add_regress_comps(self, comps):
        self._regress.update(comps)

    def get_comps(self, part: Union[Literal["backbone"], Literal["classify_branch"], Literal["regress_branch"]]):
        if part == "backbone":
            return self._backbone
        elif part == "classify_branch":
            return self._classify
        elif part == "regress_branch":
            return self._regress
        else:
            return {}


class SetCompRequiresGradMixin:
    def model_part_requires_grad(
        self,
        part: Union[Literal["backbone"], Literal["classify_branch"], Literal["regress_branch"]],
        flag: bool
    ):
        comps = self.component_parts.get_comps(part)
        if comps is None:
            return
        for part in comps.values():
            part.requires_grad_(flag)
