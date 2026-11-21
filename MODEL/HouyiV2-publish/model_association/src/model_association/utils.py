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

import ast
import collections
import datetime
import logging
import numbers
import os
import pickle
import re
import socket
import traceback
from dataclasses import asdict
from numbers import Number
from pathlib import Path
from typing import Optional, Tuple, Union, Set, TypeVar, Type, List, Generator, Iterable

import numba
import numpy as np
import numpy.typing as npt
import torch.distributed
from pydantic import BaseModel, ConfigDict


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


LOG_PATTERN = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s*\[\s*([^|]+?)\s*\|\s*([^]]+?)\s*]\s*(.*)$'
)


def extract_log_messages(
    log: str,
    levels: Union[str, List[str], None] = None
) -> Generator[Tuple[datetime.datetime, str, str]]:
    target_levels = None
    if levels is not None:
        if isinstance(levels, str):
            target_levels = {levels.upper()}
        else:
            target_levels = {level.upper() for level in levels}

    last_entry = None
    for line in log.splitlines():
        match = LOG_PATTERN.match(line)

        if match:
            if last_entry is not None:
                if target_levels is None or last_entry[1] in target_levels:
                    yield last_entry

            time_str, name, level, message = match.groups()
            try:
                dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                last_entry = (dt, level.strip(), message.strip())
            except ValueError:
                continue

        else:
            if last_entry is not None and line.strip():
                # Handle the cases where the tailing linebreak in the last entry is missing
                inner_match = LOG_PATTERN.search(line)
                if inner_match:
                    split_index = inner_match.start()
                    dt_prev, level_prev, msg_prev = last_entry
                    last_entry = (dt_prev, level_prev, f"{msg_prev}\n{line[:split_index]}")

                    if target_levels is None or last_entry[1] in target_levels:
                        yield last_entry

                    time_str, name, level, message = inner_match.groups()
                    try:
                        dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                        last_entry = (dt, level.strip(), message.strip())
                    except ValueError:
                        last_entry = None

                else:
                    dt, level, message = last_entry
                    last_entry = (dt, level, f"{message}\n{line}")

    if last_entry is not None:
        if target_levels is None or last_entry[1] in target_levels:
            yield last_entry


def get_host_ip():
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    return ip_addr


def create_dir(*dir_path_segments):
    """
    Concatenate all segments in :attr:`dir_path_segments` as a directory path
    and create the directory (and its parents) if it does not exist.

    Return:
        Concatenated directory path.

    """
    full_dir_path = Path(*dir_path_segments)
    full_dir_path.mkdir(parents=True, exist_ok=True)
    return full_dir_path


class ModuleMappingUnpickler(pickle.Unpickler):
    def __init__(self, module_mapping: dict[str, str], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.module_mapping = module_mapping

    def find_class(self, module_name: str, __global_name):
        if module_name in self.module_mapping:
            module_name = self.module_mapping[module_name]

        return super().find_class(module_name, __global_name)


def bit_mask_compress(
    array: npt.NDArray[bool],
    dtype: np.dtype,
    axis: int = -1
) -> np.ndarray:
    array = np.swapaxes(array, axis, 0)

    item_bits: int = dtype.itemsize * 8
    compressed = np.zeros(
        [int(np.ceil(array.shape[0] / item_bits)),
         *array.shape[1:]],
        dtype=dtype
    )

    _bit_mask_compress_impl(array, compressed, item_bits)

    return compressed.swapaxes(axis, 0)


@numba.njit()
def _bit_mask_compress_impl(
    array: npt.NDArray[bool],
    compressed: npt.NDArray[float],
    item_bits: int
):
    for i_row in numba.prange(len(array)):
        row = array[i_row]
        div, rem = np.divmod(i_row, item_bits)
        compressed[div] |= (array[i_row] << rem)


def bit_mask_extract(
    array: np.ndarray,
    item_bits: Optional[int] = None,
    axis: int = -1
) -> npt.NDArray[bool]:
    if not item_bits:
        item_bits = array.dtype.itemsize * 8
    elif item_bits > array.dtype.itemsize * 8:
        raise ValueError(
            "There are not enough bits in items of input `array`. "
            f"At least {item_bits} bits is expected, but there are only "
            f"{array.dtype.itemsize * 8} bits."
        )

    array = np.swapaxes(array, axis, 0)
    extracted = np.zeros(
        [item_bits * array.shape[0],
         *array.shape[1:]],
        dtype=bool
    )

    _bit_mask_extract_impl(array, extracted, item_bits)

    return extracted.swapaxes(axis, 0)


@numba.njit()
def _bit_mask_extract_impl(
    array: npt.NDArray[float],
    extracted: npt.NDArray[bool],
    item_bits: int
):
    for i_row in numba.prange(extracted.shape[0]):
        div, rem = np.divmod(i_row, item_bits)
        extracted[i_row] = array[div] & (1 << rem)


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


def setup_ddp(
    rank: int,
    world_size: int,
    local_rank: int = None,
    master_addr: str = "127.0.0.1",
    master_port: int = 29501
):
    """
    Setup distributed data parallel training environment.
    This function can only run ONCE during the whole running process.

    """
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    dist.init_process_group(
        "nccl",
        init_method="env://",
        rank=rank, world_size=world_size
    )

    if not dist.is_initialized():
        raise RuntimeError("Process group is not initialized!")

    if local_rank is None:
        local_rank = rank
    torch.cuda.set_device(local_rank)

    dist.barrier()
    torch.cuda.synchronize()


def cleanup_ddp():
    """ Destroy distributed data parallel training environment. """
    import torch.distributed as dist
    dist.destroy_process_group()


def parse_ddp_env():
    def _int_or_none(v):
        if v is not None:
            return int(v)
        return None

    rank = _int_or_none(os.environ.get("SLURM_PROCID"))
    world_size = _int_or_none(os.environ.get("SLURM_NTASKS"))
    local_rank = _int_or_none(os.environ.get("SLURM_LOCALID"))
    num_nodes = _int_or_none(os.environ.get("SLURM_NNODES"))

    master_addr = os.environ.get("MASTER_ADDR")
    master_port = _int_or_none(os.environ.get("MASTER_PORT"))

    return rank, world_size, local_rank, master_addr, master_port, num_nodes


def unwrap_ddp_model(model):
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def unwrap_state_dict(state_dict: collections.OrderedDict) -> collections.OrderedDict:
    unwrapped = collections.OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        unwrapped[k] = v
    return unwrapped


def create_dataloader(dataset, rank, world_size, ddp_mode=False, shuffle=True, **dataloader_kwargs):
    from torch.utils.data import DistributedSampler
    from torch.utils.data import RandomSampler
    from torch.utils.data import SequentialSampler
    from torch.utils.data import DataLoader

    if ddp_mode:
        sampler = DistributedSampler(
            dataset,
            # num_replicas=world_size,
            # rank=rank,
            shuffle=shuffle
        )
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        **dataloader_kwargs
    )

    return dataloader, sampler


def ddp_reduce(tensor, op: torch.distributed.ReduceOp):
    import torch.distributed
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=op)
    return tensor


def parse_nested_float_tuple(parameter: str) -> tuple[tuple[Number, Number], ...]:
    import typer

    try:
        parsed_value = ast.literal_eval(parameter)
    except (ValueError, SyntaxError):
        raise typer.BadParameter(
            f"Invalid format of parameter: {parameter}. {traceback.format_exc()}"
        )

    if len(parsed_value) > 0:
        for item in parsed_value:
            if len(item) != 2:
                raise typer.BadParameter(
                    f"Item of input parameter is not a 2-tuple: {item}"
                )

            if not (isinstance(item[0], numbers.Real) and isinstance(item[1], numbers.Real)):
                raise typer.BadParameter(
                    f"Elements in items of input parameter is not of real number "
                    f"type: {item}"
                )

    parsed_value = tuple(
        (float(item[0]), float(item[1]))
        for item in parsed_value
    )

    return parsed_value


class ParentProcessChecker:
    def __init__(self):
        self.parent_pid = os.getppid()

    def parent_proc_ok(self) -> bool:
        if os.name == "nt":  # Windows
            import psutil
            if not psutil.pid_exists(self.parent_pid):
                return False

        else:
            if os.getppid() != self.parent_pid:
                return False

        return True


def is_file1_newer_than_file2(file1, file2):
    mtime_1 = os.path.getmtime(file1)
    mtime_2 = os.path.getmtime(file2)
    return mtime_1 > mtime_2


def get_tensor_array_pair(
    data: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, np.ndarray]:
    if isinstance(data, torch.Tensor):
        if device is not None:
            data_tn = data.to(device)
        else:
            data_tn = data

        data_arr = data_tn.cpu().numpy()

    else:
        data_arr = np.asanyarray(data)
        data_tn = torch.tensor(data_arr, device=device)

    return data_tn, data_arr


def normalize(a):
    a_min = a.min()
    a_max = a.max()
    return (a - a_min) / (a_max - a_min)


def get_num_trainable_parameters(model) -> Tuple[int, int]:
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable_params, all_param = model.get_nb_trainable_parameters()
    else:
        trainable_params, all_param = 0, 0
        for param in model.parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

    return trainable_params, all_param


def check_lora_dir(
    dir: Path,
    files_to_check: Set[str] = frozenset(
        ["adapter_model.safetensors", "adapter_config.json"]
    )
) -> bool:
    """
    Check whether `dir` is a valid LoRA directory, i.e. containing files
    listed in `files_to_check`.

    """
    files_exist = {k: False for k in files_to_check}
    for f in dir.iterdir():
        if not f.is_file():
            continue

        if f.name in files_exist:
            files_exist[f.name] = True

    return all(files_exist.values())


TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


class DataclassMixin(BaseModel):
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    @classmethod
    def from_dataclass(cls: Type[TBaseModel], dataclass_obj: object) -> TBaseModel:
        return cls.model_validate(dataclass_obj)

    def to_dataclass(self, dataclass: Type) -> object:
        return dataclass(**asdict(self))


def move_nan_points_to_tail(
    batch: torch.Tensor,
    related_tensors: Optional[Iterable[torch.Tensor]] = None
) -> Tuple[torch.Tensor, Optional[Iterable[torch.Tensor]]]:
    shape = batch.shape
    if related_tensors is not None:
        for i, tensor in enumerate(related_tensors):
            if shape[:-1] != tensor.shape[:-1]:
                raise ValueError(
                    f"The shape of tensor {i} in `related_tensors` ({tensor.shape}) "
                    f"is not consistent with the tensor `batch` ({shape})"
                )

    has_nan = batch.isnan().any(-1)
    indices = torch.argsort(has_nan.long(), dim=-1, stable=True)

    if related_tensors is None:
        related_tensors_reordered = None
    else:
        related_tensors_reordered = [
            torch.gather(tensor, dim=-2, index=indices.unsqueeze(-1).expand_as(tensor))
            for tensor in related_tensors
        ]

    return (torch.gather(batch, dim=-2, index=indices.unsqueeze(-1).expand_as(batch)),
            related_tensors_reordered)


def inspect_gradients(model) -> Optional[Tuple[float, float, float, float, float]]:
    parameters = [p for p in model.parameters() if p.grad is not None]

    if len(parameters) == 0:
        return

    total_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2).item()
    total_param_norm = torch.norm(torch.stack([torch.norm(p.detach(), 2) for p in parameters]), 2).item()
    max_grad = torch.max(torch.stack([torch.max(p.grad.detach()) for p in parameters])).item()
    min_grad = torch.min(torch.stack([torch.min(p.grad.detach()) for p in parameters])).item()
    mean_grad = torch.mean(torch.stack([torch.mean(p.grad.detach()) for p in parameters])).item()

    grad_param_ratio = total_grad_norm / (total_param_norm + 1e-8)

    return total_grad_norm, max_grad, min_grad, mean_grad, grad_param_ratio
