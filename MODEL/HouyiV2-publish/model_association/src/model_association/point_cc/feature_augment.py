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

from collections import OrderedDict
from typing import Union, Tuple, Sequence, Optional, Iterable

import numpy as np
import torch


def _sine_augment_np(feature: np.ndarray, freqs, norm_to_01: bool) -> np.ndarray:
    freqs = np.reshape(freqs, [-1])
    sin_aug = (np.sin(feature[..., None] * freqs * 2 * np.pi))  # shape = feature.shape + n_freqs
    cos_aug = (np.cos(feature[..., None] * freqs * 2 * np.pi))  # shape = feature.shape + n_freqs
    augmented = np.stack([sin_aug, cos_aug], axis=-1)
    augmented = np.reshape(augmented, feature.shape + (-1,))

    if norm_to_01:
        augmented = (augmented + 1) / 2
    return augmented  # shape=(feature.shape, n_freqs * 2)


def _sine_augment_torch(feature: torch.Tensor, freqs, norm_to_01: bool) -> torch.Tensor:
    freqs = torch.tensor(freqs).reshape([-1]).to(feature.device)
    sin_aug = (torch.sin(feature.unsqueeze(-1) * freqs * 2 * np.pi))  # shape = (*feature.shape, n_freqs)
    cos_aug = (torch.cos(feature.unsqueeze(-1) * freqs * 2 * np.pi))  # shape = (*feature.shape, n_freqs)
    augmented = torch.stack([sin_aug, cos_aug], dim=-1)
    augmented = torch.reshape(augmented, feature.shape + (-1,))

    if norm_to_01:
        augmented = (augmented + 1) / 2
    return augmented  # shape=(*feature.shape, n_freqs * 2)


def sine_augment(feature: Union[np.ndarray, torch.Tensor], freqs, norm_to_01: bool = False):
    if isinstance(freqs, torch.Tensor):
        freqs_arr = freqs.detach().double().cpu().numpy().reshape([-1])
    else:
        freqs_arr = np.asarray(freqs, dtype=np.float64).reshape([-1])
    if np.all(freqs_arr == np.round(freqs_arr)):
        # Integer frequencies make sin/cos exactly 1-periodic, i.e. enc(x) ==
        # enc(frac(x)), so wrapping out-of-range values is numerically a no-op
        # and keeps the encoding exactly recoverable modulo 1. Out-of-range
        # values appear in the labels of the route's NEXT point (its time may
        # exceed the sample window / its coords may fall outside the area
        # box); the absolute value of such labels is carried by the raw label
        # channels (see FeatureAugmenter.recover's `integer_hint`).
        if isinstance(feature, torch.Tensor):
            feature = feature - torch.floor(feature)
        else:
            feature = feature - np.floor(feature)
    else:
        # With non-integer frequencies the encoding is not 1-periodic, so
        # out-of-range values would be aliased and become unrecoverable.
        if ((feature < 0) | (1 < feature)).any():
            raise ValueError(
                "Feature to be augmented must take value in range [0, 1] "
                "when the augmentation frequencies are not all integers."
            )
    if isinstance(feature, torch.Tensor):
        return _sine_augment_torch(feature, freqs, norm_to_01)
    return _sine_augment_np(feature, freqs, norm_to_01)


def _recover_augmented_feature_np(aug_feature: np.ndarray,
                                  freqs,
                                  n_samples: int,
                                  normed_to_01: bool) -> np.ndarray:
    freqs = np.reshape(freqs, [-1])

    if normed_to_01:
        aug_feature = aug_feature * 2 - 1

    x_ticks = np.arange(n_samples) / n_samples
    sin_emb_vec = np.sin(x_ticks[..., None] * freqs * 2 * np.pi)  # shape=(n_samples, n_freqs)
    cos_emb_vec = np.cos(x_ticks[..., None] * freqs * 2 * np.pi)  # shape=(n_samples, n_freqs)
    emb_vec = np.stack([sin_emb_vec, cos_emb_vec], axis=-1)
    emb_vec = np.reshape(emb_vec, (n_samples, -1)).T  # shape=(n_freqs * 2, n_samples)

    recovered = np.argmax(aug_feature @ emb_vec, axis=-1) / n_samples
    return recovered


def _recover_augmented_feature_fast_np(aug_feature: np.ndarray,
                                       freqs,
                                       normed_to_01: bool) -> np.ndarray:
    freqs = np.reshape(freqs, [-1])

    if normed_to_01:
        aug_feature = aug_feature * 2 - 1

    *batch_shape, n_aug = aug_feature.shape
    n_elements = int(np.prod(batch_shape))
    aug_flat = aug_feature.reshape(n_elements, -1)

    n_freqs = len(freqs)
    aug_reshaped = aug_flat.reshape(n_elements, n_freqs, 2)
    sin_comp = aug_reshaped[:, :, 0]
    cos_comp = aug_reshaped[:, :, 1]

    phases = np.arctan2(sin_comp, cos_comp)
    f_0 = freqs[0]
    phi_0 = phases[:, 0]

    n_candidates = int(np.ceil(f_0)) + 1
    k = np.arange(n_candidates, dtype=np.float64)

    candidates = (phi_0[:, None] + 2 * np.pi * k[None, :]) / (2 * np.pi * f_0)
    candidates = candidates - np.floor(candidates)

    max_mem_mb = 500
    bytes_per_element = 8
    elements_per_chunk = max_mem_mb * 1024 * 1024 / (bytes_per_element * n_candidates * n_freqs * 2)
    chunk_size = max(100, int(elements_per_chunk))

    best_scores = np.full(n_elements, -np.inf, dtype=np.float64)
    best_candidates = np.zeros(n_elements, dtype=np.float64)

    for start in range(0, n_elements, chunk_size):
        end = min(start + chunk_size, n_elements)
        chunk_candidates = candidates[start:end]
        chunk_aug = aug_flat[start:end]

        chunk_candidates_exp = chunk_candidates[:, :, None]
        template_sin = np.sin(2 * np.pi * freqs[None, None, :] * chunk_candidates_exp)
        template_cos = np.cos(2 * np.pi * freqs[None, None, :] * chunk_candidates_exp)
        template = np.stack([template_sin, template_cos], axis=-1).reshape(end - start, n_candidates, -1)
        scores = np.sum(chunk_aug[:, None, :] * template, axis=-1)

        best_idx = np.argmax(scores, axis=-1)
        best_scores[start:end] = scores[np.arange(end - start), best_idx]
        best_candidates[start:end] = chunk_candidates[np.arange(end - start), best_idx]

    return best_candidates.reshape(batch_shape)


def _recover_augmented_feature_torch(aug_feature: torch.Tensor,
                                     freqs,
                                     n_samples: int,
                                     normed_to_01: bool) -> torch.Tensor:
    freqs = torch.tensor(freqs).reshape([-1]).to(aug_feature.device)

    if normed_to_01:
        aug_feature = aug_feature * 2 - 1

    x_ticks = (torch.arange(n_samples)
               .to(device=aug_feature.device, dtype=aug_feature.dtype)) / n_samples
    sin_emb_vec = torch.sin(x_ticks.unsqueeze(-1) * freqs * 2 * np.pi)  # shape=(n_samples, n_freqs)
    cos_emb_vec = torch.cos(x_ticks.unsqueeze(-1) * freqs * 2 * np.pi)  # shape=(n_samples, n_freqs)
    emb_vec = torch.stack([sin_emb_vec, cos_emb_vec], dim=-1)
    emb_vec = torch.reshape(emb_vec, (n_samples, -1)).T  # shape=(n_freqs * 2, n_samples)

    recovered = torch.argmax(aug_feature @ emb_vec, dim=-1) / n_samples
    return recovered


def _recover_augmented_feature_fast_torch(aug_feature: torch.Tensor,
                                          freqs: torch.Tensor,
                                          normed_to_01: bool) -> torch.Tensor:
    freqs = torch.tensor(freqs).reshape([-1]).to(aug_feature.device, dtype=aug_feature.dtype)

    if normed_to_01:
        aug_feature = aug_feature * 2 - 1

    *batch_shape, n_aug = aug_feature.shape
    n_elements = int(np.prod(batch_shape))
    aug_flat = aug_feature.reshape(n_elements, -1)

    n_freqs = len(freqs)
    aug_reshaped = aug_flat.reshape(n_elements, n_freqs, 2)
    sin_comp = aug_reshaped[:, :, 0]
    cos_comp = aug_reshaped[:, :, 1]

    phases = torch.atan2(sin_comp, cos_comp)
    f_0 = freqs[0]
    phi_0 = phases[:, 0]

    n_candidates = int(torch.ceil(f_0).item()) + 1
    k = torch.arange(n_candidates, device=aug_feature.device, dtype=aug_feature.dtype)

    candidates = (phi_0[:, None] + 2 * np.pi * k[None, :]) / (2 * np.pi * f_0)
    candidates = candidates - torch.floor(candidates)

    chunk_size = max(100, 10000 // n_candidates)

    best_scores = torch.full((n_elements,), -np.inf, device=aug_feature.device)
    best_candidates = torch.zeros(n_elements, device=aug_feature.device)

    for start in range(0, n_elements, chunk_size):
        end = min(start + chunk_size, n_elements)
        chunk_candidates = candidates[start:end]
        chunk_aug = aug_flat[start:end]

        chunk_candidates_exp = chunk_candidates[:, :, None]
        template_sin = torch.sin(2 * np.pi * freqs[None, None, :] * chunk_candidates_exp)
        template_cos = torch.cos(2 * np.pi * freqs[None, None, :] * chunk_candidates_exp)
        template = torch.stack([template_sin, template_cos], dim=-1).reshape(end - start, n_candidates, -1)
        scores = torch.sum(chunk_aug[:, None, :] * template, dim=-1)

        best_idx = torch.argmax(scores, dim=-1).cpu().numpy()
        best_scores[start:end] = scores[np.arange(end - start), best_idx]
        best_candidates[start:end] = chunk_candidates[np.arange(end - start), best_idx]

    return best_candidates.reshape(batch_shape)


def _recover_augmented_feature(aug_feature: Union[np.ndarray, torch.Tensor],
                               freqs,
                               n_samples: int = 1000,
                               normed_to_01: bool = False):
    if isinstance(aug_feature, torch.Tensor):
        return _recover_augmented_feature_torch(aug_feature, freqs, n_samples, normed_to_01)
    return _recover_augmented_feature_np(aug_feature, freqs, n_samples, normed_to_01)


def _recover_augmented_feature_fast(aug_feature: Union[np.ndarray, torch.Tensor],
                                    freqs,
                                    normed_to_01: bool = False):
    import warnings
    warnings.warn(
        "This function requires each augmented feature pair "
        "(s_i=sin(2 pi f_i x), c_i=cos(2 pi f_i x)) satisfying s_i^2+c_i^2=1"
    )
    if isinstance(aug_feature, torch.Tensor):
        return _recover_augmented_feature_fast_torch(aug_feature, freqs, normed_to_01)
    return _recover_augmented_feature_fast_np(aug_feature, freqs, normed_to_01)


def recover_augmented_feature(aug_feature: Union[np.ndarray, torch.Tensor],
                              freqs,
                              normed_to_01: bool = False):
    return _recover_augmented_feature(aug_feature, freqs=freqs, normed_to_01=normed_to_01)


class FeatureAugmenter:
    def __init__(
        self,
        feature_indices: Tuple[int, ...],
        augment_freqs: OrderedDict[int, Sequence[float]],
        norm_to_01: bool = True
    ):
        """
        Initialize the feature augmenter.

        Args:
            feature_indices: Indices (IDs) of columns of feature data array
                input to :py:method:`FeatureAugmenter.augment`.
            augment_freqs: Mapping from index of feature to be augmented to
                its augmentation frequencies. Only those feature indices
                contained in :attr:`feature_indices` will be considered.

        """
        self.feature_indices = feature_indices
        self.augment_freqs = OrderedDict(
            [(f_idx, freqs) for f_idx, freqs in augment_freqs.items()
             if f_idx in feature_indices]
        )

        if len(self.augment_freqs) == 0:
            raise ValueError(
                "Frequency dictionary of features to be augmented is empty."
            )

        self._feat_to_aug_in_inputs = {
            f_idx: feature_indices.index(f_idx)
            for f_idx in augment_freqs.keys()
        }  # Indices of features to be augmented in un-augmented input data.
        self._aug_feat_end_indices = dict(zip(
            augment_freqs.keys(),
            np.cumsum([len(freq) for freq in augment_freqs.values()]) * 2  # sine and cosine
        ))
        self._aug_feat_start_indices = dict(zip(
            augment_freqs.keys(),
            [0] + list(self._aug_feat_end_indices.values())[:-1]
        ))
        self._n_aug_features = list(self._aug_feat_end_indices.values())[-1]

        self.norm_to_01 = norm_to_01

    @property
    def n_aug_features(self) -> int:
        """
        Number of columns augmented from features listed in
        :py:attr:`self.augment_freqs.keys()`.

        """
        return self._n_aug_features

    def augment(self, features):
        if features.shape[-1] != len(self.feature_indices):
            raise ValueError(
                f"The number of features of input data ({features.shape[-1]}) "
                f"doesn't match that of self.feature_indices "
                f"({len(self.feature_indices)})."
            )

        augmented = [sine_augment(
            features[..., self._feat_to_aug_in_inputs[f_idx]],
            self.augment_freqs[f_idx],
            norm_to_01=self.norm_to_01
        ) for f_idx in self.augment_freqs.keys()]
        if isinstance(features, torch.Tensor):
            augmented = torch.concatenate(augmented, dim=-1)
        else:
            augmented = np.concatenate(augmented, axis=-1)

        assert augmented.shape[-1] == self._n_aug_features
        return augmented

    def recover(
        self,
        augmented_features,
        indices_of_feat_to_recover: Optional[Iterable[int]] = None,
        integer_hint=None
    ):
        """
        Recover the (original) feature values from the augmented ones.

        The augmentation with integer frequencies is 1-periodic, hence the
        recovered value is the true value modulo 1, i.e. in [0, 1). When the
        original values may be out of [0, 1] (e.g. labels of the route's
        NEXT point), pass a *coarse* prediction of the absolute values
        (e.g. the model's non-augmented regression outputs) as
        :attr:`integer_hint`. The integer part is then taken from the hint:
        ``recovered + round(integer_hint - recovered)``, while the precise
        fractional part still comes from :attr:`augmented_features`.
        The hint only needs to be accurate up to 0.5 (of the normalized
        feature unit) to recover the correct integer part. For values within
        [0, 1) this reduces to the plain recovery.
        """
        if augmented_features.shape[-1] != self._n_aug_features:
            raise ValueError(
                f"The number of features of augmented data "
                f"({augmented_features.shape[-1]}) "
                f"doesn't match sum(2 * lengths of augmented feature frequencies) "
                f"({self._n_aug_features})."
            )

        if indices_of_feat_to_recover is None:
            indices_of_feat_to_recover = self.augment_freqs.keys()
        else:
            indices_of_feat_to_recover = tuple(indices_of_feat_to_recover)

        recovered = [
            recover_augmented_feature(
                augmented_features[
                ...,
                self._aug_feat_start_indices[f_idx]
                :self._aug_feat_end_indices[f_idx]
                ],
                self.augment_freqs[f_idx],
                normed_to_01=self.norm_to_01
            )
            for f_idx in indices_of_feat_to_recover
        ]
        if isinstance(recovered[0] if len(recovered) > 0 else augmented_features,
                      torch.Tensor):
            recovered = (
                torch.stack(recovered, dim=-1)
                if len(recovered) > 0 else
                torch.zeros(augmented_features.shape[:-1] + (0,),
                            dtype=augmented_features.dtype,
                            device=augmented_features.device)
            )
        else:
            recovered = (
                np.stack(recovered, axis=-1)
                if len(recovered) > 0 else
                np.zeros(augmented_features.shape[:-1] + (0,),
                         dtype=augmented_features.dtype)
            )

        assert recovered.shape[-1] == len(indices_of_feat_to_recover)
        if integer_hint is not None:
            if integer_hint.shape != recovered.shape:
                raise ValueError(
                    f"`integer_hint` has shape {integer_hint.shape}, but "
                    f"expected {recovered.shape} (the shape of recovered "
                    f"features)."
                )
            if isinstance(recovered, torch.Tensor):
                integer_part = torch.round(
                    integer_hint.to(recovered.dtype) - recovered
                )
            else:
                integer_part = np.round(integer_hint - recovered)
            recovered = recovered + integer_part
        return recovered
