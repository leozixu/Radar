#  Copyright 2025 NKLRDS AI team
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

"""
Module of data queues.

Authors: Junxiang Xiao, Yuanli Feng.

"""

import abc
import logging
import re
from pathlib import Path
from typing import Optional
import numpy as np
from model_association.utils import setup_logger

from model_association import radar_definition as rd


class DataQueueBase(abc.ABC):
    @abc.abstractmethod
    def get(self, *args, **kwargs):
        """
        Get one pack of data from the queue.

        """
        raise NotImplementedError


class DataTimeDelayedDataCache:
    def __init__(self, time_delay: float):
        self.time_delay = time_delay
        self._start_time = None
        self._end_time = None
        self._delayed_cache = None
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def consume(self, data) -> Optional[np.ndarray]:
        input_times = data[:, rd.PointFeatureIndex.RADAR_TIME.value]

        if self._start_time is None:
            self._start_time = np.min(input_times)
            self._end_time = np.max(input_times)
            self._delayed_cache = data
        else:
            self._delayed_cache = np.concatenate(
                (self._delayed_cache, data),
                axis=0
            )
            self._end_time = max(self._end_time, np.max(input_times))

        if (self._end_time - self._start_time >= self.time_delay): 
            self._enabled = False
            return self._delayed_cache

        return None


class AllLocalFilesVisited(Exception):
    pass


class LocalFileDataQueue(DataQueueBase):
    """
    A data queue based on local record files.

    """

    def __init__(
        self,
        data_dir: Path,
        point_batch_time_range: float = 1,
        start_time: float = 0,
        end_time: float = 0,
        data_length: int = 64,
        data_file_name_regex: Optional[str] = r".*\.csv",
        log_level=logging.INFO,
        log_file_path: Optional[Path] = None
    ):
        if not data_dir.is_dir():
            raise ValueError(
                f"Data directory path {data_dir} does not point to a valid "
                f"directory."
            )
        if start_time is not None:
            self.start_time = start_time
        if end_time is not None:
            self.end_time = end_time
        self.data_dir = data_dir
        self.data_files: list[Path] = [
            f for f in self.data_dir.iterdir()
            if (not data_file_name_regex) or re.match(data_file_name_regex, f.name)
        ]

        if len(self.data_files) == 0:
            raise ValueError(
                "There is no data file "
                + (f"whose name matches pattern '{data_file_name_regex}'"
                   if data_file_name_regex else '')
                + f"in {self.data_dir}"
            )

        self.data_length = data_length

        self.point_batch_time_range = point_batch_time_range
        self._current_file_index = 0
        self._current_file_data = None
        self._batch_ranges = None
        self._next_batch_index = None
        self.step_count = 0

        self.logger = setup_logger(
            self.__class__.__name__,
            log_level=log_level,
            log_file_name=log_file_path
        )

    def get(self, *args, **kwargs):
        track_state_reset_args = None
        if self._current_file_data is None:  # if new file is to be read, ...
            # tell the caller to reset state of tracker manager
            track_state_reset_args = (
                self.data_files[self._current_file_index - 1]
                if self._current_file_index >= 1
                else None,
            )

            if self._current_file_index == len(self.data_files):  # if there is no unvisited data files
                return AllLocalFilesVisited(), track_state_reset_args

            # if there is still unvisited data files, read data from it
            _data_file = self.current_file
            current_file_data = np.loadtxt(_data_file, delimiter=',', skiprows=1)

            rows, cols = current_file_data.shape
            if cols < 64:
                # 扩展的列数都填 0，扩展到 64 维
                pad = np.zeros((rows, 64 - cols))
                current_file_data = np.hstack((current_file_data, pad))

            current_file_data[:, rd.PointFeatureIndex.MODEL_CLASS:] = 0
            current_file_data[:, rd.PointFeatureIndex.POINT_ID] = np.arange(1, current_file_data.shape[0] + 1)
            if hasattr(self, 'start_time'):
                current_file_data = current_file_data[
                    current_file_data[:, rd.PointFeatureIndex.RADAR_TIME] >= self.start_time]
            if hasattr(self, 'end_time'):
                current_file_data = current_file_data[
                    current_file_data[:, rd.PointFeatureIndex.RADAR_TIME] <= self.end_time]
            data_length = max(self.data_length, current_file_data.shape[1])
            curr_length = min(self.data_length, current_file_data.shape[1])
            self._current_file_data = np.zeros((len(current_file_data), data_length), dtype=current_file_data.dtype)
            self._current_file_data[:, :curr_length] = current_file_data[:, :curr_length]

            self.logger.info(
                "Read data file: %s  Data shape: %s",
                _data_file, self._current_file_data.shape
            )
            self._current_file_data = self._current_file_data[
                np.argsort(self._current_file_data[:, rd.PointFeatureIndex.RADAR_TIME.value])
            ]
            # RADAR_TIME 为原始编码(ms), point_batch_time_range 同为 ms
            _batch_ids = (
                (self._current_file_data[:, rd.PointFeatureIndex.RADAR_TIME.value]
                 - self._current_file_data[0, rd.PointFeatureIndex.RADAR_TIME.value])
                // self.point_batch_time_range
            )
            _batch_id_set = np.unique(_batch_ids)
            _borders = np.concatenate(
                ([0],
                 np.where(np.diff(_batch_ids))[0] + 1,
                 [len(self._current_file_data)])
            )
            self._batch_ranges = list(zip(_borders[:-1], _borders[1:]))
            self._next_batch_index = 0

            self.logger.debug(
                f"%d batches of points found in file.",
                len(self._batch_ranges)
            )
        _batch_range = self._batch_ranges[self._next_batch_index]
        data_to_return = self._current_file_data[_batch_range[0]:_batch_range[1]]
        self.logger.debug(
            "Point batch %d/%d returned.",
            self._next_batch_index, len(self._batch_ranges)
        )

        self._next_batch_index += 1
        if self._next_batch_index == len(self._batch_ranges):
            self._current_file_index += 1
            self._current_file_data = None
            self._batch_ranges = None
            self._next_batch_index = None

        data_to_return[:, rd.PointFeatureIndex.TRACK_ID_ALGO] = 0
        data_to_return[:, rd.PointFeatureIndex.PACK_ID] = self.step_count
        self.step_count += 1
        return data_to_return, track_state_reset_args

    @property
    def current_file(self):
        return self.data_files[self._current_file_index]

class TimeDelayedLocalFileDataQueue(LocalFileDataQueue):
    def __init__(
        self,
        data_dir: Path,
        time_step: float = 1,
        time_delay: float = 0,
        data_length: int = 64,
        data_file_name_regex: Optional[str] = r".*\.dat",
        start_time: float = 0,
        end_time: float = 0,
        log_level: int = logging.INFO,
        log_file_path: Optional[Path] = None
    ):

        super().__init__(
            data_dir, time_step, start_time, end_time, data_length=data_length,
            data_file_name_regex=data_file_name_regex,
            log_level=log_level, log_file_path=log_file_path
        )

        # 缓存内比较的是原始编码(ms), time_delay 也按 ms 传入
        self._delayed_cache = DataTimeDelayedDataCache(time_delay)

    def get(self, *args, **kwargs):
        raw_data, state_reset_args = super().get()

        if self._delayed_cache is None or not self._delayed_cache.enabled:
            return raw_data, state_reset_args

        data_returned = self._delayed_cache.consume(raw_data)
        if data_returned is None:
            return None, state_reset_args

        return data_returned, state_reset_args

