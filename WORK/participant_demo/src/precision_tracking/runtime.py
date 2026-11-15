"""Load the paired-return profile and process one protocol scan at a time."""

from dataclasses import asdict, dataclass
from math import isfinite

import tracker_compat

from . import causal_birth_paths, clock_association, emission_identity, paired_returns, pair_swap_probe
from . import raw_timing_detection, teacher_assoc
from .registered_pair_recovery import BothRecoveries


FEATURE_SCHEMAS = {
    "teacher": teacher_assoc.FEATURES,
    "pair_orientation": pair_swap_probe.PAIR_NAMES,
    "birth": causal_birth_paths.FEATURES,
    "scan_clock": clock_association.FEATURES,
    "pair_signature": paired_returns.FEATURES,
    "raw_return": raw_timing_detection.FEATURES,
    "quality": tracker_compat.QUALITY_FEATURES,
    "emission_detection": emission_identity.FEATURES,
    "emission_ownership": emission_identity.FEATURES,
}
TRUSTED = emission_identity.FEATURES.index("trusted_proposals")


@dataclass(frozen=True, slots=True)
class PrecisionSettings:
    association_floor: float = .035
    association_margin: float = .30
    birth_floor: float = .86
    pair_start_floor: float = .985
    pair_link_floor: float = .95
    pair_fallback_floor: float = .96
    raw_floor: float = .025
    quality_threshold: float = .16
    emission_detection_floor: float = .25
    emission_raw_floor: float = .80

    def __post_init__(self):
        thresholds = asdict(self)
        margin = thresholds.pop("association_margin")
        if not isfinite(margin) or margin < 0:
            raise ValueError("The association margin must be finite and nonnegative")
        if not all(isfinite(value) and 0 < value < 1 for value in thresholds.values()):
            raise ValueError("Precision tracking thresholds must be between zero and one")
        if self.pair_link_floor > self.pair_start_floor:
            raise ValueError("The pair link threshold must not exceed the startup threshold")


@dataclass(frozen=True, slots=True)
class PrecisionRuntime:
    settings: PrecisionSettings
    models: dict
    metadata: dict

    def new_packet(self):
        config, models = self.settings, self.models
        pairing = BothRecoveries(models["pair_signature"], config.pair_start_floor, config.pair_link_floor,
                                 early=True, wide=True, fallback_floor=config.pair_fallback_floor)
        raw_gate = raw_timing_detection.RawReturnGate(models["raw_return"], config.raw_floor)
        tracker = causal_birth_paths.BirthOnline(
            models["teacher"], models["pair_orientation"], models["birth"], models["scan_clock"],
            config.birth_floor, "supplement", paired_birth=True, joint=True, two_hit_pairs=True,
            pairing=pairing, raw_gate=raw_gate, association_floor=config.association_floor,
            association_margin=config.association_margin,
        )
        return tracker, emission_identity.EmissionFeatures()

    def process_circle(self, points, state):
        packet = state.get("_precision_packet")
        if packet is None:
            packet = state["_precision_packet"] = self.new_packet()
        tracker, history = packet
        proposals, quality = tracker_compat.decisions_with_features(points, state, tracker.track)
        if not proposals:
            history.build(points, proposals, quality, (), state, self.models["scan_clock"])
            return []
        model, config = self.models, self.settings
        legacy = model["quality"].predict(quality, num_threads=1)
        features = history.build(points, proposals, quality, legacy, state, model["scan_clock"])
        detection = model["emission_detection"].predict(features, num_threads=1)
        ownership = model["emission_ownership"].predict(features, num_threads=1)
        timing = emission_identity.current_raw_probabilities(points, proposals, state)
        keep = emission_identity.accepted_emissions(
            legacy >= config.quality_threshold, detection, ownership, features[:, TRUSTED] < 3,
            (0., config.emission_detection_floor, 0.), timing, config.emission_raw_floor,
        )
        return [proposal for proposal, accepted in zip(proposals, keep) if accepted]


def load_precision(profile, settings, descriptors, feature_version, file_hash, load_model):
    settings = PrecisionSettings(**settings)
    models, info = {}, {}
    for name, features in FEATURE_SCHEMAS.items():
        models[name], info[name] = load_model(profile, descriptors.get(name), features, name)
        if models[name] is None:
            raise ValueError(f"Missing precision tracking model {name}: {profile}")
    return PrecisionRuntime(settings, models, {
        "backend": "precision", "profile": profile.name, "profile_sha256": file_hash(profile),
        "feature_version": feature_version, "models": info, "settings": asdict(settings),
    })
