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

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Optional, TypeVar

import numba
from pydantic import BaseModel


def setup_logger(
    logger_name: str,
    log_level=logging.DEBUG,
    log_file_name: Optional[Path] = None,
    console_output: bool = True
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    log_format = logging.Formatter(
        "%(asctime)s [ %(name)s | %(levelname)s ] %(message)s"
    )

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    if log_file_name is not None:
        log_file_name.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setFormatter(log_format)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    logger.setLevel(log_level)

    return logger


def dict_to_numba(
    d: dict,
    key_type,
    value_type
) -> numba.typed.Dict:
    """ Convert python dict object into numba typed dictionary. """
    d_nb = numba.typed.Dict.empty(
        key_type=key_type,
        value_type=value_type
    )
    for k, v in d.items():
        d_nb[k] = v
    return d_nb


def unwrap_state_dict(state_dict: collections.OrderedDict) -> collections.OrderedDict:
    unwrapped = collections.OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        unwrapped[k] = v
    return unwrapped


TBaseModel = TypeVar("TBaseModel", bound=BaseModel)
