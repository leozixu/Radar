"""Entrypoint for the original TCP client and explicit experiment profiles."""

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from tracker import ASSOCIATION_FEATURES, BIRTH_FEATURES, TrackerModels, TrackerSettings, process_candidates


DEFAULT_PROFILE = Path(__file__).resolve().parents[1] / "models" / "frozen.json"
FEATURE_VERSION = 1


@dataclass(frozen=True, slots=True)
class Runtime:
    settings: TrackerSettings
    models: TrackerModels
    metadata: dict

    def process_circle(self, points, state):
        return process_candidates(points, state, self.settings, self.models)


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(values):
    settings = TrackerSettings(**values)
    for variances in (settings.measurement, settings.process):
        if np.shape(variances) != (3,) or not np.all(np.isfinite(variances)) or np.any(np.array(variances) <= 0):
            raise ValueError("Measurement and process variances must each contain three finite positive values")
    return settings


def _load_model(profile, descriptor, features, kind):
    if descriptor is None:
        return None, None
    from lightgbm import Booster

    if isinstance(descriptor, str):
        name, expected_hash = descriptor, None
    else:
        name, expected_hash = descriptor["file"], descriptor.get("sha256")
    path = (profile.parent / name).resolve(strict=True)
    digest = _file_hash(path)
    if expected_hash is not None and digest != expected_hash.lower():
        raise ValueError(f"{kind} model hash mismatch: {path}")
    model = Booster(model_file=str(path))
    if tuple(model.feature_name()) != features:
        raise ValueError(f"{kind} model feature schema disagrees with tracker: {path}")
    return model, {"file": name, "sha256": digest}


@lru_cache(maxsize=8)
def _read_runtime(profile):
    document = json.loads(profile.read_text(encoding="utf-8-sig"))
    if document.get("feature_version", FEATURE_VERSION) != FEATURE_VERSION:
        raise ValueError(f"Unsupported tracker feature version: {profile}")
    predictors = document.get("models", {})
    backend = document.get("backend", "learned")
    if backend == "precision":
        from precision_tracking.runtime import load_precision

        return load_precision(profile, document.get("settings", {}), predictors,
                              FEATURE_VERSION, _file_hash, _load_model)
    if backend == "compat":
        from tracker_compat import CompatibilityRuntime, QUALITY_FEATURES

        threshold = document.get("settings", {}).get("threshold", 0.20)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Compatibility quality threshold must be within [0, 1]")
        model, info = _load_model(profile, predictors.get("quality"), QUALITY_FEATURES, "quality")
        if model is None:
            raise ValueError("The compatibility profile requires its trained quality model")
        return CompatibilityRuntime(model, threshold, {
            "backend": backend, "profile": profile.name, "profile_sha256": _file_hash(profile),
            "feature_version": FEATURE_VERSION, "models": {"quality": info},
            "settings": {"threshold": threshold},
        })
    if backend != "learned":
        raise ValueError(f"Unsupported tracker backend: {backend}")
    names = {"association": ASSOCIATION_FEATURES, "birth": BIRTH_FEATURES}
    if profile == DEFAULT_PROFILE and any(predictors.get(kind) is None for kind in names):
        raise ValueError("The frozen profile must include both trained models")
    settings = _settings(document.get("settings", {}))
    models, model_info = {}, {}
    for kind, features in names.items():
        models[kind], model_info[kind] = _load_model(profile, predictors.get(kind), features, kind)
    settings_dict = asdict(settings)
    settings_dict["measurement"] = list(settings.measurement)
    settings_dict["process"] = list(settings.process)
    return Runtime(settings, TrackerModels(**models), {
        "backend": backend, "profile": profile.name, "profile_sha256": _file_hash(profile),
        "feature_version": FEATURE_VERSION, "models": model_info, "settings": settings_dict,
    })


def load_runtime(config_path=None):
    profile = DEFAULT_PROFILE if config_path is None else Path(config_path).resolve()
    return _read_runtime(profile)


def process_circle(points, state):
    return load_runtime().process_circle(points, state)
