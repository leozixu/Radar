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

"""
LoRA (PEFT) helpers

"""

import json
from pathlib import Path
from typing import Optional, Any

import torch


# A parameter name (as returned by ``PeftModel.named_parameters``) is mapped to
# one of the three model parts.  LoRA adapters are injected only onto the
# backbone, while the two heads are stored (and trained) as ``modules_to_save``.
def _lora_param_part(name: str) -> Optional[str]:
    """Return the part ("backbone"/"regress_branch"/"classify_branch") that a
    parameter belongs to in a PEFT/LoRA model, or None if it does not belong to
    a trainable part (e.g. an ``original_module`` we never touch).

    ``PeftModel`` prefixes every parameter with ``base_model.model.``; the
    backbone LoRA adapters come from ``<module>.lora_A/lora_B`` and the
    fine-tuned heads are stored under ``<head>.modules_to_save.default``.
    """
    if ".lora_" in name:
        return "backbone"
    if "regressor" in name and "modules_to_save" in name:
        return "regress_branch"
    if ("classifier" in name or "point_classifier" in name) and "modules_to_save" in name:
        return "classify_branch"
    return None


def set_lora_part_requires_grad(model, part: str, flag: bool) -> None:
    """Set ``requires_grad`` on every parameter of ``model`` that belongs to
    ``part`` (see :func:`_lora_param_part`).

    Unlike ``model_part_requires_grad`` (which operates on the raw modules and
    only reaches the ``original_module`` of a PEFT model, i.e. the weights we
    never train), this helper addresses the *active* trainable parameters: the
    backbone ``lora_A/lora_B`` adapters and the ``modules_to_save.default``
    head copies.  It is a no-op for non-PEFT models (no parameter matches).
    """
    for name, p in model.named_parameters():
        if _lora_param_part(name) == part and p.requires_grad != flag:
            p.requires_grad_(flag)


def _adapter_state_dict(lora_weight_path):
    """Load the raw state dict of a saved LoRA adapter.

    The adapter is saved by PEFT as ``adapter_model.safetensors`` (or the legacy
    ``adapter_model.bin``).  Uses ``safetensors`` when available.
    """
    p = Path(lora_weight_path)
    safetensors_path = p / "adapter_model.safetensors"
    bin_path = p / "adapter_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))
    if bin_path.is_file():
        loaded = torch.load(bin_path, map_location="cpu", weights_only=True)
        return loaded if isinstance(loaded, dict) else loaded.state_dict()
    raise FileNotFoundError(
        f"No adapter weights (adapter_model.safetensors / .bin) found under "
        f"{lora_weight_path}."
    )


def _adapter_config(lora_weight_path) -> Any:
    """Load the ``adapter_config.json`` of a saved LoRA adapter."""
    p = Path(lora_weight_path)
    cfg_path = p / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"No adapter_config.json found under {lora_weight_path}."
        )
    with open(cfg_path) as f:
        return json.load(f)


def _normalize_peft_key(name: str) -> str:
    """Canonicalize the name of a PEFT parameter so that adapter-saved keys can
    be compared to the in-memory model's keys.

    A saved adapter stores:
      * backbone LoRA as ``...self.key.lora_A.weight``; the in-memory model
        names it ``...self.key.lora_A.default.weight``;
      * fine-tuned heads as ``...point_classifier.0.weight``; the in-memory
        model names them ``...point_classifier.modules_to_save.default.0.weight``.

    We strip the ``.default.`` (LoRA adapter index) and the
    ``.modules_to_save.default`` (head copy) segments so both sides collapse to
    the same canonical key.
    """
    norm = name.replace(".default.", ".")
    norm = norm.replace("modules_to_save.default.", "")
    norm = norm.replace(".modules_to_save.", ".")
    return norm


def report_lora_mount(model, lora_weight_path, logger) -> None:
    """Log which parts were successfully mounted from the LoRA checkpoint.

    Only logs information; it does not raise.  Semantic validation is done
    separately by :func:`validate_lora_mount`.
    """
    try:
        adapter_sd = _adapter_state_dict(lora_weight_path)
    except Exception as e:
        logger.warning(f"Could not load LoRA adapter state dict for reporting: {e}")
        return
    adapter_config = _adapter_config(lora_weight_path)

    # Backbone LoRA adapters from the adapter config.
    target_modules = adapter_config.get("target_modules", [])
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    # module prefixes that got a LoRA adapter in the checkpoint.
    lora_prefixes = set()
    for k in adapter_sd:
        if ".lora_A" in k or ".lora_B" in k:
            nk = _normalize_peft_key(k)
            # "base_model.model.bert_encoder...self.query.lora_A.weight" -> "query"
            seg = nk.split("base_model.model.")[-1]
            parts = seg.split(".")
            # find the last module part before ".lora_"
            for i, seg_i in enumerate(parts):
                if seg_i == "lora_A" or seg_i == "lora_B":
                    lora_prefixes.add(parts[i - 1])
                    break

    mounted_heads = []
    for mod in adapter_config.get("modules_to_save", []):
        if any(f".{mod}." in _normalize_peft_key(k) for k in adapter_sd):
            mounted_heads.append(mod)

    missing_targets = [m for m in target_modules if m not in lora_prefixes]
    logger.info(
        f"LoRA mount from {lora_weight_path}: backbone LoRA adapters on "
        f"target_modules={sorted(lora_prefixes)} "
        f"(expected={target_modules}, missing={missing_targets}); "
        f"fine-tuned heads mounted={mounted_heads} "
        f"(from modules_to_save={adapter_config.get('modules_to_save', [])})."
    )


def validate_lora_mount(model, lora_weight_path, logger) -> None:
    """Validate that the LoRA checkpoint is compatible with the model.

    Only the parts that appear in *both* the adapter and the in-memory model
    are checked: for each such part (a backbone ``target_modules`` LoRA adapter
    or a fine-tuned head in ``modules_to_save``) every parameter shape must be
    identical so that the adapter can be injected into (backbone) or fully
    loaded onto (heads) the current model.  A shape conflict raises.

    Parts that are present in the adapter but absent from the model, or present
    in the model but absent from the adapter, are ignored -- this mirrors the
    PEFT loading behaviour, which silently skips such weights.
    """
    adapter_sd = _adapter_state_dict(lora_weight_path)
    adapter_cfg = _adapter_config(lora_weight_path)

    # Map model key -> tensor, keyed by the *canonical* (normalized) name so that
    # adapter keys (which omit the ``.default.`` / ``modules_to_save.default``
    # segments) can be matched to the in-memory model.
    model_by_norm = {
        _normalize_peft_key(k): v for k, v in model.state_dict().items()
    }

    def _check_common(part: str, keys_in_adapter) -> None:
        for k in keys_in_adapter:
            nk = _normalize_peft_key(k)
            if nk not in model_by_norm:
                # Part exists only in the adapter; PEFT would skip it.
                continue
            if tuple(adapter_sd[k].shape) != tuple(model_by_norm[nk].shape):
                raise ValueError(
                    f"LoRA parameter shape mismatch for {k} ({part}): adapter "
                    f"{tuple(adapter_sd[k].shape)} vs model "
                    f"{tuple(model_by_norm[nk].shape)}."
                )

    # --- Backbone target_modules ------------------------------------------ #
    # For each LoRA module the adapter carries, if the model has the same LoRA
    # module, its parameter shapes must match (i.e. it is injectable).
    target_modules = adapter_cfg.get("target_modules", [])
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    for mod in target_modules:
        lora_keys = [
            k for k in adapter_sd
            if f".{mod}." in k and (".lora_A" in k or ".lora_B" in k)
        ]
        _check_common(f"backbone target_module '{mod}'", lora_keys)

    # --- Heads (modules_to_save) ------------------------------------------ #
    # For each fine-tuned head the adapter carries, if the model has the same
    # head, its parameter shapes must match (i.e. it can be fully loaded).
    modules_to_save = adapter_cfg.get("modules_to_save", [])
    if isinstance(modules_to_save, str):
        modules_to_save = [modules_to_save]
    for head in modules_to_save:
        head_keys = [k for k in adapter_sd if f".{head}." in k]
        _check_common(f"head '{head}'", head_keys)
