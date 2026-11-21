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
Script for running apps.

Author: Junxiang Xiao.

"""
import json
import inspect
import logging
import traceback
from pathlib import Path
from typing import Annotated, Optional

import click
import torch
import typer
from model_association.utils import setup_logger

from apps import PlaybackTrackerApp


app = typer.Typer(no_args_is_help=True)


@app.callback()
def main():
    """Houyi inference server."""


@app.command()
def run_playback_tracker(
    data_dir: Annotated[
        str, typer.Argument(
            help="Data directory."
        )
    ],
    model_weight_path: Annotated[
        str, typer.Argument(
            help="Path to saved model weight."
        )
    ],
    lora_weight_path: Annotated[Optional[str], typer.Option(
        help="Path to the directory of LoRA adapter weights (containing "
             "adapter_config.json). If provided, these weights will be "
             "injected into the base model specified by 'model_path'."
    )] = None,
    num_point_features: Annotated[
        int, typer.Option(
            help="Number of features of each point."
        )
    ] = 72,
    log_level: Annotated[
        str, typer.Option(
            help="Logging level.",
            click_type=click.Choice([
                "CRITICAL",
                "FATAL",
                "ERROR",
                "WARNING",
                "WARN",
                "INFO",
                "DEBUG",
                "NOTSET"
            ]),
            case_sensitive=False
        )
    ] = "DEBUG",
    proc_config_path: Annotated[
        str, typer.Option(
            help="Path of the processing configuration."
        )
    ]="../settings/proc_config.json",
    save_dir: Annotated[
        Optional[str], typer.Option(
            help="Directory to saved track data."
        )
    ] = None
):
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        log_file_path = save_dir / "playback_tracker.log"
    else:
        log_file_path = None

    logger = setup_logger(
        inspect.currentframe().f_code.co_name,
        log_level,
        log_file_name=log_file_path
    )
    with open(proc_config_path) as f:
        process_config = json.load(f)
    tracker_app = PlaybackTrackerApp(
        Path(data_dir),
        num_point_features,
        Path(model_weight_path),
        device=torch.device("cuda"),
        proc_params=process_config,
        lora_weight_path=Path(lora_weight_path) if lora_weight_path is not None else None,
        log_level=log_level,
        log_file_path=log_file_path,
        save_dir=save_dir
    )

    try:
        tracker_app.run()
    except (KeyboardInterrupt, InterruptedError):
        logger.info("Tracker app is interrupted.")
    except Exception as e:
        logger.error(
            "An error is raised in app running: %s. %s",
            e, traceback.format_exc()
        )

if __name__ == "__main__":
    app()
