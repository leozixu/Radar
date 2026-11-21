# -*- coding: utf-8 -*-

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

from typing import Optional, Union

import torch
import torch.nn as nn

from model_association.point_cc.config import CCModelTrainingLossTermCoeffs


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, smooth=1e-5):
        target = target.float()
        intersect = torch.sum(pred * target)

        union = torch.sum(target * target) + torch.sum(pred * pred)

        dice_score = (2. * intersect + smooth) / (union + smooth)
        loss = 1 - dice_score

        return loss


class SymmetricDiceLoss(nn.Module):
    def __init__(self):
        super(SymmetricDiceLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, smooth=1e-5):
        target = target.float()

        # Foreground
        intersect_pos = (pred * target).sum()
        union_pos = pred.square().sum() + target.square().sum()
        dice_pos = (2. * intersect_pos + smooth) / (union_pos + smooth)

        # Background
        intersect_neg = ((1 - pred) * (1 - target)).sum()
        union_neg = (1 - pred).square().sum() + (1 - target).square().sum()
        dice_neg = (2. * intersect_neg + smooth) / (union_neg + smooth)

        dice_score = (dice_pos + dice_neg) * 0.5
        return 1 - dice_score


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha: Union[float, torch.Tensor, None] = 0.25,  # shape=(C,)
                 gamma: float = 2.0,
                 reduction: str = "mean",
                 pos_weight: Optional[torch.Tensor] = None,
                 from_logits: bool = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: Predicted output, shape=(N,) (binary classification) or
                (N, C, ...) (multiple(C>1)/binary(C=1) classification).
            targets: Target values, shape=(N,) (binary classification) or
                (N, ...) (multiple classification indices) or
                (N, C, ...) (boolean one-hot)

        Returns:
            Single value of focal loss.

        Examples:
            # Binary classification
            >>> input = torch.randn(3, requires_grad=True)
            >>> target = torch.rand(3, requires_grad=False) > 0.5
            >>> loss_fn = FocalLoss(from_logits=True)
            >>> loss = loss_fn(input, target)

            # Multiple classification
            >>> n_classes = 5
            >>> input = torch.randn(3, n_classes, requires_grad=True)
            >>> target = torch.nn.functional.one_hot(torch.randint(0, n_classes, (3,), requires_grad=False), n_classes)
            >>> loss_fn = FocalLoss(alpha=torch.rand(n_classes), from_logits=True)
            >>> loss = loss_fn(input, target)

        """
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(1)  # shape=(N, 1)
        # inputs.shape=(N, C, ...)

        if targets.ndim == inputs.ndim - 1:
            if inputs.size(1) == 1:  # binary classification
                targets = targets.unsqueeze(1)
            else:  # multiple classification
                # `targets` are classification indices, convert them into one-hot vectors
                targets = torch.nn.functional.one_hot(
                    targets, num_classes=inputs.size(-1)
                )
        # targets.shape=(N, C, ...)

        if self.from_logits:
            if inputs.size(1) == 1:  # shape=(N, C=1, ...)
                # Binary classification
                probs = torch.sigmoid(inputs)
            else:
                probs = torch.softmax(inputs, dim=1)
        else:
            probs = inputs
        targets_float = targets.float()

        if probs.size(1) == 1:
            p_t = torch.where(targets_float == 1, probs, 1 - probs)

            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.to(device=probs.device)
                alpha_t = torch.where(targets_float == 1, alpha, 1 - alpha)
            elif self.alpha is None:
                alpha_t = torch.ones_like(p_t)
            else:
                alpha_t = torch.where(targets_float == 1, self.alpha, 1 - self.alpha)

            focal_weight = alpha_t * (1 - p_t) ** self.gamma
            if self.from_logits:
                bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    inputs, targets_float, pos_weight=self.pos_weight, reduction="none"
                )
            else:
                bce_loss = torch.nn.functional.binary_cross_entropy(
                    probs, targets_float, reduction="none"
                )
            focal_loss = focal_weight * bce_loss
        else:
            p_t = torch.sum(probs * targets_float, dim=1)  # shape=(N, ...)

            if isinstance(self.alpha, torch.Tensor):
                alpha_t = torch.einsum("c,nc...->n...", self.alpha, targets_float)  # shape=(N, ...)
            elif self.alpha is None:
                alpha_t = torch.ones_like(p_t)
            else:
                raise ValueError(
                    "In multiple classification case, `alpha` must be a tensor "
                    f"of shape (num_classes={inputs.size(1)},), but {self.alpha} is given."
                )

            focal_weight = alpha_t * (1 - p_t) ** self.gamma
            ce_loss = torch.nn.functional.cross_entropy(
                probs,
                torch.argmax(targets_float, dim=1),
                reduction="none"
            )
            focal_loss = focal_weight * ce_loss

        if self.reduction in (None, 'none'):
            return focal_loss
        elif hasattr(focal_loss, self.reduction):
            return getattr(focal_loss, self.reduction)()
        raise ValueError(f"Cannot call reduction '{self.reduction}' on focal loss tensor.")


class CCLoss(nn.Module):
    def __init__(
        self,
        loss_coeffs: CCModelTrainingLossTermCoeffs,
        device="cpu",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.focal_loss_fn = FocalLoss(alpha=0.75, gamma=2.0, reduction="mean", from_logits=False)
        self.dice_loss_fn = DiceLoss()
        self.l1_loss_fn = torch.nn.L1Loss()
        self.circle_loss_fn = DiceLoss()
        self.l1_dice_fn = BalancedL1Loss()

        self.uncertainty_weighting = loss_coeffs.use_uncertainty_weighting
        self.log_vars = nn.Parameter(-torch.log(torch.tensor(
            [loss_coeffs.classify_coeff, loss_coeffs.regress_coeff, loss_coeffs.circle_coeff],
            device=device
        )))  # 2log(sigma)=-log(coeff)
        self.log_vars.requires_grad_(self.uncertainty_weighting)
        if not self.uncertainty_weighting:
            self._coeffs = torch.exp(-self.log_vars)

    @property
    def coeffs(self) -> torch.Tensor:
        if not self.uncertainty_weighting:
            return self._coeffs
        return torch.exp(-self.log_vars)  # 1/sigma^2

    @property
    def classify_coeff(self) -> float:
        return self.coeffs[0].item()

    @property
    def regress_coeff(self) -> float:
        return self.coeffs[1].item()

    @property
    def circle_coeff(self) -> float:
        return self.coeffs[2].item()

    def forward(
        self,
        labels: torch.Tensor,
        prob_preds=None,
        near_region_points_mask=None,
        probs_circle_pred=None,
        labels_circle=None,
        reg_preds=None,
        reg_labels=None
    ):
        device = labels.device
        if prob_preds is not None:
            l_classify = (
                5 * self.focal_loss_fn(prob_preds, labels)
                + self.dice_loss_fn(prob_preds, labels)
            )

            if near_region_points_mask is not None:
                l_classify += 5 * self.dice_loss_fn(prob_preds[near_region_points_mask],
                                                    labels.float()[near_region_points_mask])
        else:
            l_classify = torch.tensor(0.0, device=device)

        if probs_circle_pred is not None and labels_circle is not None:
            l_circle = self.circle_loss_fn(probs_circle_pred, labels_circle.float())
        else:
            l_circle = torch.tensor(0.0, device=device)

        if reg_preds is not None and reg_labels is not None:
            # The NEXT point label of a route without a next point (or of a
            # padded point) is NaN; such label positions must be masked out of
            # the regression loss, otherwise the (mean) L1 losses become NaN.
            reg_finite_mask = torch.isfinite(reg_labels)
            reg_label_mask = labels.bool().unsqueeze(-1) & reg_finite_mask
            if reg_finite_mask.any():
                l1_loss = self.l1_loss_fn(
                    reg_preds[reg_finite_mask], reg_labels[reg_finite_mask].float())
            else:
                l1_loss = torch.tensor(0.0, device=device)
            l1_dice = 5 * self.l1_dice_fn(
                reg_preds, reg_labels.float(), reg_label_mask)
            l_regress = (l1_loss + l1_dice)
        else:
            l_regress = torch.tensor(0.0, device=device)

        l_classify *= self.coeffs[0]
        l_regress *= self.coeffs[1]
        l_circle *= self.coeffs[2]
        loss = l_classify + l_regress + l_circle
        if self.uncertainty_weighting:
            loss += self.log_vars.sum() * 0.5  # normalization term

        return loss, {"classify": l_classify,
                      "regress": l_regress,
                      "circle": l_circle}


def cross_route_penalty(
    corr_mat: torch.Tensor,
    true_route_ids: torch.Tensor
):
    """
    Evaluate penalty based on correlation values between points from different
    routes.

    Args:
        corr_mat: shape=(batch_size, n_points, n_points)
        true_route_ids: shape=(batch_size, n_points)

    Returns:
        Penalty value.

    """
    # Mask for correlations between any point and an NaN padding point
    nan_points_mask = true_route_ids.isnan()
    corr_nan_points_mask = nan_points_mask.unsqueeze(1) | nan_points_mask.unsqueeze(2)
    # shape=(batch_size, n_points, n_points)

    # Mask for correlations between any point and a false point
    false_points_mask = true_route_ids < 0
    corr_false_points_mask = false_points_mask.unsqueeze(1) | false_points_mask.unsqueeze(2)
    # shape=(batch_size, n_points, n_points)

    # Mask for correlations between any points whose route IDs are different
    # or between one false point and another point
    cross_route_mask = (
        (true_route_ids.unsqueeze(1) != true_route_ids.unsqueeze(2))
        | corr_false_points_mask
    )
    # shape=(batch_size, n_points, n_points)

    # Filter out NaN points
    cross_route_mask = cross_route_mask & (~corr_nan_points_mask)

    penalty = (
        (cross_route_mask * corr_mat).abs().square().view(-1).sum()
        / cross_route_mask.long().view(-1).sum()
    )
    return penalty


class BalancedL1Loss(nn.Module):
    def __init__(self, a_t=5., a_f=2.):
        super().__init__()
        self.a_t = a_t
        self.a_f = a_f

    def forward(self, y_pred, y_true, mask):
        smooth = 1e-5

        # Only entries with a finite target are considered (the target of a
        # route's NEXT point is NaN when the route has no such point).
        finite = torch.isfinite(y_true)
        true_mask = mask & finite
        false_mask = (~mask) & finite

        if true_mask.sum() != 0:
            l1_t = torch.nn.functional.l1_loss(
                input=y_pred[true_mask],
                target=y_true[true_mask].float()
            )
        else:
            l1_t = torch.tensor(0., device=y_pred.device, dtype=y_pred.dtype)

        if false_mask.sum() != 0:
            l1_f = torch.nn.functional.l1_loss(
                input=y_pred[false_mask],
                target=y_true[false_mask].float()
            )
        else:
            l1_f = torch.tensor(0., device=y_pred.device, dtype=y_pred.dtype)

        # return (l1_t * l1_f + smooth) / (self.a_t * l1_t + self.a_f * l1_f + smooth)
        return l1_t * (self.a_t * l1_t + self.a_f * l1_f)
